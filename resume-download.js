/**
 * 浏览器端模型断点续传：
 * - 完整文件仍由 Transformers.js 写入 Cache API
 * - 未下完的大文件进度写入 IndexedDB，中断/刷新后用 HTTP Range 续传
 * - 劫持 fetch，对 HF / hf-mirror 的 resolve URL 生效
 */

const DB_NAME = "doyen-resume-v1";
const STORE = "partials";
const PERSIST_EVERY = 256 * 1024; // 每 256KB 落盘
const MIN_RESUME_BYTES = 64 * 1024;

let installed = false;
let nativeFetch = null;
let resumeListeners = [];
/** @type {Map<string, { flush: () => Promise<void> }>} */
const activeDownloads = new Map();
let lifecycleBound = false;

function notifyResume(info) {
  resumeListeners.forEach(function (fn) {
    try {
      fn(info);
    } catch (_) {}
  });
}

export function onResumeProgress(fn) {
  if (typeof fn !== "function") return function () {};
  resumeListeners.push(fn);
  return function () {
    resumeListeners = resumeListeners.filter(function (x) {
      return x !== fn;
    });
  };
}

function openDb() {
  return new Promise(function (resolve, reject) {
    if (typeof indexedDB === "undefined") {
      reject(new Error("no indexedDB"));
      return;
    }
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = function () {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "url" });
      }
    };
    req.onsuccess = function () {
      resolve(req.result);
    };
    req.onerror = function () {
      reject(req.error || new Error("idb open failed"));
    };
  });
}

function idbReq(req) {
  return new Promise(function (resolve, reject) {
    req.onsuccess = function () {
      resolve(req.result);
    };
    req.onerror = function () {
      reject(req.error);
    };
  });
}

async function getPartial(url) {
  try {
    const db = await openDb();
    try {
      return await idbReq(
        db.transaction(STORE, "readonly").objectStore(STORE).get(url)
      );
    } finally {
      db.close();
    }
  } catch (_) {
    return null;
  }
}

async function putPartial(record) {
  try {
    const db = await openDb();
    try {
      await idbReq(
        db.transaction(STORE, "readwrite").objectStore(STORE).put(record)
      );
    } finally {
      db.close();
    }
  } catch (err) {
    console.warn("[doyen-resume] save partial failed", err);
  }
}

async function deletePartial(url) {
  try {
    const db = await openDb();
    try {
      await idbReq(
        db.transaction(STORE, "readwrite").objectStore(STORE).delete(url)
      );
    } finally {
      db.close();
    }
  } catch (_) {}
}

export async function clearResumePartials() {
  try {
    const db = await openDb();
    try {
      await idbReq(
        db.transaction(STORE, "readwrite").objectStore(STORE).clear()
      );
    } finally {
      db.close();
    }
  } catch (_) {
    try {
      indexedDB.deleteDatabase(DB_NAME);
    } catch (_) {}
  }
}

export async function listResumePartials() {
  try {
    const db = await openDb();
    try {
      const rows = await idbReq(
        db.transaction(STORE, "readonly").objectStore(STORE).getAll()
      );
      return (rows || []).map(function (r) {
        const size =
          r && r.blob && typeof r.blob.size === "number" ? r.blob.size : 0;
        return {
          url: r.url,
          size: size,
          total: r.total || 0,
          etag: r.etag || "",
        };
      });
    } finally {
      db.close();
    }
  } catch (_) {
    return [];
  }
}

/** 是否有未下完的半成品 */
export async function hasIncompleteDownloads() {
  const rows = await listResumePartials();
  return rows.some(function (r) {
    return r && r.size > 0 && (!r.total || r.size < r.total);
  });
}

/**
 * 估算续传总进度（已知体积的半成品）
 * @returns {Promise<{pct:number, loaded:number, total:number, files:number}>}
 */
export async function estimateResumeProgress() {
  const rows = await listResumePartials();
  let loaded = 0;
  let total = 0;
  let files = 0;
  rows.forEach(function (r) {
    if (!r || !r.size) return;
    files += 1;
    loaded += r.size;
    total += r.total > 0 ? r.total : r.size;
  });
  const pct = total > 0 ? Math.min(99, (loaded / total) * 100) : 0;
  return { pct: pct, loaded: loaded, total: total, files: files };
}

/** 把进行中的下载立刻落盘（切后台 / 刷新前调用） */
export async function flushActiveDownloads() {
  const tasks = [];
  activeDownloads.forEach(function (entry) {
    if (entry && typeof entry.flush === "function") {
      tasks.push(
        Promise.resolve()
          .then(function () {
            return entry.flush();
          })
          .catch(function () {})
      );
    }
  });
  if (!tasks.length) return;
  await Promise.all(tasks);
}

function bindLifecycleFlush() {
  if (lifecycleBound || typeof window === "undefined") return;
  lifecycleBound = true;
  const kick = function () {
    flushActiveDownloads().catch(function () {});
  };
  window.addEventListener("pagehide", kick);
  window.addEventListener("beforeunload", kick);
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") kick();
  });
}

function requestUrl(input) {
  if (typeof input === "string") return input;
  if (input && typeof input.url === "string") return input.url;
  try {
    return String(input);
  } catch (_) {
    return "";
  }
}

function isResumableModelUrl(url) {
  if (!url || url.indexOf("http") !== 0) return false;
  if (!/huggingface\.co|hf-mirror\.com/i.test(url)) return false;
  if (!/\/resolve\//i.test(url)) return false;
  if (/\.onnx(\?|$)/i.test(url)) return true;
  if (/tokenizer\.json(\?|$)/i.test(url)) return true;
  if (/tokenizer_config\.json(\?|$)/i.test(url)) return true;
  return false;
}

function parseContentRangeTotal(header) {
  if (!header) return 0;
  const m = /\/(\d+)\s*$/.exec(header);
  return m ? parseInt(m[1], 10) : 0;
}

function fileNameFromUrl(url) {
  try {
    const u = new URL(url);
    const parts = u.pathname.split("/");
    return parts[parts.length - 1] || "file";
  } catch (_) {
    return "file";
  }
}

/**
 * 返回 status=200 的 Response：先吐出已下载部分，再续传网络剩余，并周期性写入 IDB。
 */
function wrapResumableResponse(url, networkRes, start, partialBlob, meta) {
  const total = meta.total || 0;
  const etag = meta.etag || "";
  const contentType =
    meta.contentType ||
    networkRes.headers.get("Content-Type") ||
    "application/octet-stream";

  let savedBlob = partialBlob || new Blob();
  let pending = [];
  let pendingSize = 0;
  let loaded = start;
  let persistChain = Promise.resolve();

  function flushPending(force) {
    if (!pending.length) return persistChain;
    if (!force && pendingSize < PERSIST_EVERY) return persistChain;
    const chunkBlob = new Blob(pending);
    pending = [];
    pendingSize = 0;
    savedBlob = new Blob([savedBlob, chunkBlob]);
    const snapshot = savedBlob;
    const sizeNow = snapshot.size;
    persistChain = persistChain
      .then(function () {
        return putPartial({
          url: url,
          etag: etag,
          total: total,
          blob: snapshot,
          updatedAt: Date.now(),
        });
      })
      .then(function () {
        notifyResume({
          url: url,
          file: fileNameFromUrl(url),
          loaded: sizeNow,
          total: total,
          resumed: start > 0,
          status: "persist",
        });
      })
      .catch(function () {});
    return persistChain;
  }

  activeDownloads.set(url, {
    flush: function () {
      return flushPending(true);
    },
  });

  const stream = new ReadableStream({
    start: function (controller) {
      (async function () {
        try {
          if (start > 0 && partialBlob && partialBlob.size > 0) {
            notifyResume({
              url: url,
              file: fileNameFromUrl(url),
              loaded: start,
              total: total,
              resumed: true,
              status: "resume",
            });
            const reader0 = partialBlob.stream().getReader();
            for (;;) {
              const chunk = await reader0.read();
              if (chunk.done) break;
              controller.enqueue(chunk.value);
            }
          }

          if (!networkRes.body) {
            throw new Error("empty body");
          }
          const reader = networkRes.body.getReader();
          for (;;) {
            const chunk = await reader.read();
            if (chunk.done) break;
            const value = chunk.value;
            controller.enqueue(value);
            pending.push(value);
            pendingSize += value.byteLength || value.length || 0;
            loaded += value.byteLength || value.length || 0;
            if (pendingSize >= PERSIST_EVERY) {
              await flushPending(false);
            }
          }

          await flushPending(true);
          await deletePartial(url);
          activeDownloads.delete(url);
          notifyResume({
            url: url,
            file: fileNameFromUrl(url),
            loaded: total || loaded,
            total: total || loaded,
            resumed: start > 0,
            status: "done",
          });
          controller.close();
        } catch (err) {
          try {
            await flushPending(true);
          } catch (_) {}
          activeDownloads.delete(url);
          controller.error(err);
        }
      })();
    },
    cancel: function () {
      return flushPending(true).then(function () {
        activeDownloads.delete(url);
      });
    },
  });

  const headers = new Headers();
  headers.set("Content-Type", contentType);
  if (total > 0) headers.set("Content-Length", String(total));
  if (etag) headers.set("ETag", etag);

  return new Response(stream, {
    status: 200,
    statusText: "OK",
    headers: headers,
  });
}

async function resumableFetch(url, init, fetchImpl) {
  const method = ((init && init.method) || "GET").toUpperCase();
  if (method !== "GET") {
    return fetchImpl(url, init);
  }

  let partial = await getPartial(url);
  let start =
    partial && partial.blob && partial.blob.size > 0 ? partial.blob.size : 0;

  const headers = new Headers((init && init.headers) || {});
  if (start > 0) {
    headers.set("Range", "bytes=" + start + "-");
    if (partial.etag) {
      headers.set("If-Range", partial.etag);
    }
  }

  const networkRes = await fetchImpl(
    url,
    Object.assign({}, init || {}, { headers: headers })
  );

  if (start > 0 && networkRes.status === 200) {
    await deletePartial(url);
    partial = null;
    start = 0;
  }

  if (networkRes.status !== 200 && networkRes.status !== 206) {
    return networkRes;
  }

  let total = 0;
  let etag = networkRes.headers.get("ETag") || (partial && partial.etag) || "";
  const contentType = networkRes.headers.get("Content-Type") || "";

  if (networkRes.status === 206) {
    total =
      parseContentRangeTotal(networkRes.headers.get("Content-Range")) ||
      (partial && partial.total) ||
      0;
    if (!total) {
      const cl = parseInt(networkRes.headers.get("Content-Length") || "0", 10);
      if (cl > 0) total = start + cl;
    }
  } else {
    total = parseInt(networkRes.headers.get("Content-Length") || "0", 10) || 0;
  }

  if (total > 0 && total < MIN_RESUME_BYTES && start === 0) {
    return networkRes;
  }

  if (start > 0 && total > 0 && start >= total) {
    await deletePartial(url);
    return fetchImpl(url, init);
  }

  if (start > 0) {
    notifyResume({
      url: url,
      file: fileNameFromUrl(url),
      loaded: start,
      total: total,
      resumed: true,
      status: "resume",
    });
  }

  return wrapResumableResponse(url, networkRes, start, partial && partial.blob, {
    total: total,
    etag: etag,
    contentType: contentType,
  });
}

export function installResumableFetch() {
  if (installed) return function () {};
  if (typeof window === "undefined" || typeof window.fetch !== "function") {
    return function () {};
  }
  nativeFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = requestUrl(input);
    if (!isResumableModelUrl(url)) {
      return nativeFetch(input, init);
    }
    return resumableFetch(url, init, nativeFetch);
  };
  installed = true;
  bindLifecycleFlush();
  return function uninstall() {
    if (!installed) return;
    if (nativeFetch) window.fetch = nativeFetch;
    installed = false;
    nativeFetch = null;
  };
}

export function isResumableFetchInstalled() {
  return installed;
}
