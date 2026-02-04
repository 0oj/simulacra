// ============================================================================
// IndexedDB file store — persist NDJSON files for quick reload
// ============================================================================

const DB_NAME = 'simulacra'
const DB_VERSION = 1
const STORE_NAME = 'files'
const MAX_RECENT = 20

export interface StoredFile {
  name: string
  content: string
  savedAt: number // epoch ms
  size: number
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'name' })
        store.createIndex('savedAt', 'savedAt')
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export async function saveFile(name: string, content: string): Promise<void> {
  const db = await open()
  const tx = db.transaction(STORE_NAME, 'readwrite')
  const store = tx.objectStore(STORE_NAME)

  const entry: StoredFile = {
    name,
    content,
    savedAt: Date.now(),
    size: content.length,
  }
  store.put(entry)

  // Prune old entries beyond MAX_RECENT
  const idx = store.index('savedAt')
  const countReq = idx.count()
  countReq.onsuccess = () => {
    const total = countReq.result
    if (total > MAX_RECENT) {
      let toDelete = total - MAX_RECENT
      const cursor = idx.openCursor() // ascending by savedAt
      cursor.onsuccess = () => {
        const c = cursor.result
        if (c && toDelete > 0) {
          c.delete()
          toDelete--
          c.continue()
        }
      }
    }
  }

  return new Promise((resolve, reject) => {
    tx.oncomplete = () => { db.close(); resolve() }
    tx.onerror = () => { db.close(); reject(tx.error) }
  })
}

export async function getFile(name: string): Promise<StoredFile | null> {
  const db = await open()
  const tx = db.transaction(STORE_NAME, 'readonly')
  const store = tx.objectStore(STORE_NAME)
  const req = store.get(name)

  return new Promise((resolve, reject) => {
    req.onsuccess = () => { db.close(); resolve(req.result ?? null) }
    req.onerror = () => { db.close(); reject(req.error) }
  })
}

export async function listRecent(): Promise<StoredFile[]> {
  const db = await open()
  const tx = db.transaction(STORE_NAME, 'readonly')
  const store = tx.objectStore(STORE_NAME)
  const idx = store.index('savedAt')
  const entries: StoredFile[] = []

  return new Promise((resolve, reject) => {
    const cursor = idx.openCursor(null, 'prev') // newest first
    cursor.onsuccess = () => {
      const c = cursor.result
      if (c && entries.length < MAX_RECENT) {
        // Return metadata only — omit content for the list
        const v = c.value as StoredFile
        entries.push({ name: v.name, content: '', savedAt: v.savedAt, size: v.size })
        c.continue()
      } else {
        db.close()
        resolve(entries)
      }
    }
    cursor.onerror = () => { db.close(); reject(cursor.result) }
  })
}

export async function deleteFile(name: string): Promise<void> {
  const db = await open()
  const tx = db.transaction(STORE_NAME, 'readwrite')
  tx.objectStore(STORE_NAME).delete(name)

  return new Promise((resolve, reject) => {
    tx.oncomplete = () => { db.close(); resolve() }
    tx.onerror = () => { db.close(); reject(tx.error) }
  })
}
