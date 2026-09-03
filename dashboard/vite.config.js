import { execFile } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import react from '@vitejs/plugin-react'
import { sites } from '@openai/sites-vite-plugin'
import { defineConfig } from 'vite'

const execFileAsync = promisify(execFile)
const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(here, '..')
const python = path.join(root, '.venv', 'Scripts', 'python.exe')
const payloadScript = path.join(root, 'scripts', 'dashboard_payload.py')
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/

function forecastApi() {
  return {
    name: 'weatherpicks-forecast-api',
    configureServer(server) {
      server.middlewares.use('/api/dashboard', async (req, res) => {
        try {
          const url = new URL(req.url ?? '/', 'http://localhost')
          const requestedDate = url.searchParams.get('date')
          if (requestedDate && !ISO_DATE.test(requestedDate)) {
            res.statusCode = 400
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: `Invalid date ${JSON.stringify(requestedDate)}. Use YYYY-MM-DD.` }))
            return
          }
          const args = [payloadScript]
          if (requestedDate) args.push('--date', requestedDate)
          const { stdout } = await execFileAsync(python, args, {
            cwd: root,
            windowsHide: true,
            maxBuffer: 1024 * 1024,
          })
          res.statusCode = 200
          res.setHeader('Content-Type', 'application/json')
          res.end(stdout)
        } catch (error) {
          console.error('[weatherpicks-forecast-api]', error.stderr || error.message || error)
          res.statusCode = 500
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ error: 'Forecast service unavailable. Check the dev server logs.' }))
        }
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), forecastApi(), sites()],
  server: { host: '127.0.0.1', port: 5173 },
})
