import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { History, Trash2, Eye, Calendar, Leaf, X } from 'lucide-react'
import { ScanHistory, PredictionResult } from '../types'

export default function HistorySection() {
  const [scanHistory, setScanHistory] = useState<ScanHistory[]>([])
  const [selectedScan, setSelectedScan] = useState<ScanHistory | null>(null)

  useEffect(() => {
    // Load scan history from localStorage
    const savedHistory = localStorage.getItem('plantScanHistory')
    if (savedHistory) {
      try {
        const history = JSON.parse(savedHistory)
        setScanHistory(history)
      } catch (error) {
        console.error('Error loading scan history:', error)
      }
    }
  }, [])

  const addToHistory = (result: PredictionResult, imageData: string) => {
    const newScan: ScanHistory = {
      id: Date.now().toString(),
      image: imageData,
      result,
      date: new Date().toISOString()
    }

    const updatedHistory = [newScan, ...scanHistory].slice(0, 20) // Keep last 20 scans
    setScanHistory(updatedHistory)
    localStorage.setItem('plantScanHistory', JSON.stringify(updatedHistory))
  }

  const removeFromHistory = (id: string) => {
    const updatedHistory = scanHistory.filter(scan => scan.id !== id)
    setScanHistory(updatedHistory)
    localStorage.setItem('plantScanHistory', JSON.stringify(updatedHistory))
  }

  const clearHistory = () => {
    setScanHistory([])
    localStorage.removeItem('plantScanHistory')
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60))
    
    if (diffInHours < 1) return 'Just now'
    if (diffInHours < 24) return `${diffInHours} hour${diffInHours > 1 ? 's' : ''} ago`
    if (diffInHours < 48) return 'Yesterday'
    return date.toLocaleDateString()
  }

  if (scanHistory.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="inline-block p-4 bg-soil-100 dark:bg-soil-800 rounded-full mb-4">
          <History className="w-12 h-12 text-soil-400 dark:text-soil-500" />
        </div>
        <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-2">
          No scans yet
        </h3>
        <p className="text-soil-600 dark:text-soil-400">
          Upload and analyze your first plant leaf image to see it here
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm text-soil-500 dark:text-soil-400">
            {scanHistory.length} scan{scanHistory.length !== 1 ? 's' : ''}
          </span>
        </div>
        
        <button
          onClick={clearHistory}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 transition-colors"
        >
          <Trash2 className="w-4 h-4" />
          Clear History
        </button>
      </div>

      {/* History Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <AnimatePresence>
          {scanHistory.map((scan, index) => (
            <motion.div
              key={scan.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className="bg-white dark:bg-soil-800 rounded-xl shadow-lg border border-soil-200 dark:border-soil-700 overflow-hidden hover:shadow-xl transition-shadow"
            >
              {/* Image */}
              <div className="relative h-48 bg-soil-100 dark:bg-soil-700">
                <img
                  src={scan.image}
                  alt="Plant leaf"
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-2 right-2">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    scan.result.confidence >= 90
                      ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                      : scan.result.confidence >= 70
                      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400'
                      : scan.result.confidence >= 50
                      ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
                      : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
                  }`}>
                    {scan.result.confidence}%
                  </span>
                </div>
              </div>

              {/* Content */}
              <div className="p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h4 className="font-semibold text-soil-900 dark:text-white text-sm leading-tight mb-1">
                      {scan.result.formatted_class_name}
                    </h4>
                    <p className="text-xs text-soil-500 dark:text-soil-400">
                      {scan.result.severity} confidence
                    </p>
                  </div>
                  
                  <button
                    onClick={() => removeFromHistory(scan.id)}
                    className="p-1 text-soil-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-soil-500 dark:text-soil-400">
                    <Calendar className="w-3 h-3" />
                    {formatDate(scan.date)}
                  </div>
                  
                  <button
                    onClick={() => setSelectedScan(scan)}
                    className="inline-flex items-center gap-1 px-3 py-1 text-xs bg-plant-100 dark:bg-plant-900/20 text-plant-700 dark:text-plant-300 rounded-full hover:bg-plant-200 dark:hover:bg-plant-900/40 transition-colors"
                  >
                    <Eye className="w-3 h-3" />
                    View
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Scan Detail Modal */}
      <AnimatePresence>
        {selectedScan && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedScan(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white dark:bg-soil-800 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold text-soil-900 dark:text-white">
                    Scan Details
                  </h3>
                  <button
                    onClick={() => setSelectedScan(null)}
                    className="p-2 text-soil-400 hover:text-soil-600 dark:hover:text-soil-300 transition-colors"
                  >
                    <X className="w-6 h-6" />
                  </button>
                </div>

                <div className="space-y-4">
                  <img
                    src={selectedScan.image}
                    alt="Plant leaf"
                    className="w-full h-64 object-cover rounded-lg"
                  />
                  
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium text-soil-900 dark:text-white mb-2">Result</h4>
                      <p className="text-soil-600 dark:text-soil-400 text-sm">
                        {selectedScan.result.formatted_class_name}
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-soil-900 dark:text-white mb-2">Confidence</h4>
                      <p className="text-soil-600 dark:text-soil-400 text-sm">
                        {selectedScan.result.confidence}%
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-soil-900 dark:text-white mb-2">Severity</h4>
                      <p className="text-soil-600 dark:text-soil-400 text-sm">
                        {selectedScan.result.severity}
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-soil-900 dark:text-white mb-2">Date</h4>
                      <p className="text-soil-600 dark:text-soil-400 text-sm">
                        {new Date(selectedScan.date).toLocaleString()}
                      </p>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-soil-900 dark:text-white mb-2">Recommendations</h4>
                    <div className="space-y-2 text-sm text-soil-600 dark:text-soil-400">
                      <p><strong>Treatment:</strong> {selectedScan.result.recommendations.treatment}</p>
                      <p><strong>Prevention:</strong> {selectedScan.result.recommendations.prevention}</p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
