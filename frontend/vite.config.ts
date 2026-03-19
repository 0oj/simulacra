import { defineConfig, Plugin } from 'vite'
import { resolve } from 'path'
import { existsSync, readdirSync, statSync, readFileSync } from 'fs'

// Output directory for scenario NDJSON files
// Defaults to /tmp/simulacra, can be overridden with SIMULACRA_OUTPUT env var
const outputDir = process.env.SIMULACRA_OUTPUT || '/tmp/simulacra'

// Plugin to list and serve NDJSON files from the output directory
function outputPlugin(): Plugin {
  return {
    name: 'output-plugin',
    configureServer(server) {
      // List files endpoint
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

      // Serve output files directly
      server.middlewares.use('/output', (req, res, next) => {
        const url = req.url || ''
        const filename = url.replace(/^\//, '').split('?')[0]
        const filepath = resolve(outputDir, filename)

        if (!existsSync(filepath)) {
          res.statusCode = 404
          res.end('File not found')
          return
        }

        try {
          const content = readFileSync(filepath, 'utf-8')
          res.setHeader('Content-Type', 'application/x-ndjson')
          res.end(content)
        } catch (err) {
          res.statusCode = 500
          res.end(String(err))
        }
      })
    },
  }
}

export default defineConfig({
  plugins: [outputPlugin()],
  server: {
    fs: {
      // Allow serving files from outside the project root
      allow: ['.', outputDir],
    },
    proxy: {
      // Forward simulation API calls to the Python sim server
      '/api/simulate': 'http://localhost:8767',
      '/api/models': 'http://localhost:8767',
    },
  },
})
