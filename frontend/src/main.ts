import './style.css'
import { App } from './app'
import { InteractiveApp } from './interactive'

// Check for interactive mode via URL param
const params = new URLSearchParams(window.location.search)
if (params.get('mode') === 'live') {
  new InteractiveApp()
} else {
  new App()
}
