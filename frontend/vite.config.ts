import { defineConfig, Plugin } from 'vite'
import { resolve } from 'path'
import { existsSync, readdirSync, statSync } from 'fs'

// Output directory for scenario NDJSON files
// Defaults to /tmp/simulacra, can be overridden with SIMULACRA_OUTPUT env var
const outputDir = process.env.SIMULACRA_OUTPUT || '/tmp/simulacra'

// Plugin to list NDJSON files in the output directory
function fileListPlugin(): Plugin {
  return {
    name: 'file-list',
    configureServer(server) {
      server.middlewares.use('/api/files', (req, res) => {
        res.setHeader('Content-Type', 'application/json')

        try {
          if (!existsSync(outputDir)) {
            res.end(JSON.stringify({ files: [], error: 'Output directory not found' }))
            return
          }

          const files = readdirSync(outputDir)
            .filter(f => f.endsWith('.ndjson'))
            .map(name => {
              const path = resolve(outputDir, name)
              const stats = statSync(path)
              return {
                name,
                size: stats.size,
                modified: stats.mtime.toISOString(),
              }
            })
            .sort((a, b) => b.modified.localeCompare(a.modified)) // Newest first

          res.end(JSON.stringify({ files }))
        } catch (err) {
          res.end(JSON.stringify({ files: [], error: String(err) }))
        }
      })
    },
  }
}

export default defineConfig({
  plugins: [fileListPlugin()],
  server: {
    fs: {
      // Allow serving files from outside the project root
      allow: ['.', outputDir],
    },
    proxy: {
      // Proxy /output/* to the output directory
      '/output': {
        target: 'http://localhost:5173',
        rewrite: (path) => path.replace(/^\/output/, '/@fs' + resolve(outputDir)),
      },
    },
  },
})
