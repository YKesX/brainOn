import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { WalletProvider } from './context/WalletContext.tsx'
import { EEGProvider } from './context/EEGContext.tsx'
import { Toaster } from 'react-hot-toast'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <WalletProvider>
      <EEGProvider>
        <Toaster
          position='top-right'
          toastOptions={{
            style: {
              background: '#333',
              color: '#fff',
            },
            success: {
              icon: '🚀',
            },
          }}
        />
        <App />
      </EEGProvider>
    </WalletProvider>
  </StrictMode>,
)
