// API Configuration for different environments
const API_CONFIG = {
  // Development (local)
  development: {
    baseUrl: 'http://localhost:8000'
  },
  // Production (deployed)
  production: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'https://your-backend-url.onrender.com'
  }
}

// Get current environment
const isDevelopment = process.env.NODE_ENV === 'development'

// Export the appropriate config
export const apiConfig = isDevelopment ? API_CONFIG.development : API_CONFIG.production

// Helper function to get full API URL
export const getApiUrl = (endpoint: string): string => {
  return `${apiConfig.baseUrl}${endpoint}`
}
