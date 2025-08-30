import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Camera, X, Loader2 } from 'lucide-react'
import { PredictionResult, UploadState } from '../types'
import { getApiUrl } from '../config/api'

interface ImageUploadProps {
  onAnalysisStart: () => void
  onPredictionComplete: (result: PredictionResult) => void
  isAnalyzing: boolean
}

export default function ImageUpload({ onAnalysisStart, onPredictionComplete, isAnalyzing }: ImageUploadProps) {
  const [uploadState, setUploadState] = useState<UploadState>({
    isDragging: false,
    selectedFile: null,
    preview: null,
    error: null
  })
  
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setUploadState((prev: UploadState) => ({ ...prev, isDragging: true }))
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setUploadState((prev: UploadState) => ({ ...prev, isDragging: false }))
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setUploadState((prev: UploadState) => ({ ...prev, isDragging: false }))
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [])

  const handleFileSelect = (file: File) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      setUploadState((prev: UploadState) => ({ ...prev, error: 'Please select an image file' }))
      return
    }

    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      setUploadState((prev: UploadState) => ({ ...prev, error: 'File size must be less than 10MB' }))
      return
    }

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setUploadState({
        isDragging: false,
        selectedFile: file,
        preview: e.target?.result as string,
        error: null
      })
    }
    reader.readAsDataURL(file)
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleRemoveFile = () => {
    setUploadState({
      isDragging: false,
      selectedFile: null,
      preview: null,
      error: null
    })
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleAnalyze = async () => {
    if (!uploadState.selectedFile) return

    onAnalysisStart()

    try {
      const formData = new FormData()
      formData.append('file', uploadState.selectedFile)

              const response = await fetch(getApiUrl('/predict'), {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to analyze image')
      }

      const result = await response.json()
      
      // Transform the result to match our interface
      const transformedResult: PredictionResult = {
        predicted_class: result.predicted_class,
        formatted_class_name: result.predicted_class.replace(/_/g, ' ').replace(/\s+/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
        confidence: result.confidence,
        severity: result.confidence >= 90 ? 'High' : result.confidence >= 70 ? 'Medium' : result.confidence >= 50 ? 'Low' : 'Very Low',
        top_predictions: result.top3_predictions || [],
        recommendations: result.recommendations || {
          description: 'Analysis complete',
          treatment: 'No specific treatment recommended',
          prevention: 'Continue monitoring plant health'
        },
        timestamp: new Date().toISOString()
      }

      onPredictionComplete(transformedResult)
    } catch (error) {
      console.error('Error analyzing image:', error)
      setUploadState((prev: UploadState) => ({ ...prev, error: 'Failed to analyze image. Please try again.' }))
    }
  }

  const openFileDialog = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 ${
          uploadState.isDragging
            ? 'border-plant-500 bg-plant-50 dark:bg-plant-950/20'
            : 'border-soil-300 dark:border-soil-600 hover:border-plant-400 dark:hover:border-plant-500'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInputChange}
          className="hidden"
        />

        {!uploadState.preview ? (
          <div className="space-y-4">
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              className="inline-block"
            >
              <Upload className="w-16 h-16 text-soil-400 dark:text-soil-500 mx-auto" />
            </motion.div>
            
            <div>
              <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-2">
                Drop your plant leaf image here
              </h3>
              <p className="text-soil-600 dark:text-soil-400 mb-4">
                or click to browse files
              </p>
              
              <button
                onClick={openFileDialog}
                className="inline-flex items-center gap-2 px-6 py-3 bg-plant-600 hover:bg-plant-700 text-white font-medium rounded-lg transition-colors"
              >
                <Camera className="w-5 h-5" />
                Choose Image
              </button>
            </div>
            
            <p className="text-sm text-soil-500 dark:text-soil-400">
              Supports JPG, PNG, GIF up to 10MB
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative inline-block">
              <img
                src={uploadState.preview}
                alt="Preview"
                className="max-w-full h-64 object-contain rounded-lg"
              />
              <button
                onClick={handleRemoveFile}
                className="absolute -top-2 -right-2 p-1 bg-red-500 hover:bg-red-600 text-white rounded-full transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <div>
              <p className="text-soil-600 dark:text-soil-400 mb-4">
                {uploadState.selectedFile?.name}
              </p>
              
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="inline-flex items-center gap-2 px-8 py-4 bg-plant-600 hover:bg-plant-700 disabled:bg-soil-400 text-white font-semibold rounded-lg transition-colors disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="inline-flex items-center gap-2"
                    >
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span>AI Analyzing...</span>
                    </motion.div>
                  </>
                ) : (
                  <>
                    <Camera className="w-5 h-5" />
                    Analyze Leaf
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Loading Overlay */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center"
          >
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="bg-white dark:bg-soil-800 rounded-2xl p-8 shadow-2xl max-w-md mx-4 text-center"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="w-16 h-16 mx-auto mb-4"
              >
                <div className="w-full h-full border-4 border-plant-200 border-t-plant-600 rounded-full" />
              </motion.div>
              
              <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-2">
                AI Analysis in Progress
              </h3>
              <p className="text-soil-600 dark:text-soil-400 mb-4">
                Our AI is carefully examining your plant leaf...
              </p>
              
              <div className="space-y-2">
                <motion.div
                  animate={{ width: ["0%", "100%"] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                  className="h-2 bg-plant-200 rounded-full overflow-hidden"
                >
                  <div className="h-full bg-gradient-to-r from-plant-400 to-plant-600 rounded-full" />
                </motion.div>
                <p className="text-sm text-soil-500 dark:text-soil-400">
                  Processing image data...
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Display */}
      <AnimatePresence>
        {uploadState.error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-400"
          >
            {uploadState.error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Tips */}
      <div className="bg-soil-50 dark:bg-soil-800/50 rounded-lg p-6">
        <h4 className="font-semibold text-soil-900 dark:text-white mb-3">
          💡 Tips for best results:
        </h4>
        <ul className="text-sm text-soil-600 dark:text-soil-400 space-y-1">
          <li>• Take a clear, well-lit photo of the affected leaf</li>
          <li>• Ensure the leaf fills most of the frame</li>
          <li>• Avoid shadows and reflections</li>
          <li>• Include both healthy and affected areas if possible</li>
        </ul>
      </div>
    </div>
  )
}
