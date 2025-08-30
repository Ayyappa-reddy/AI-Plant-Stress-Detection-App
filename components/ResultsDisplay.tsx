import { motion } from 'framer-motion'
import { CheckCircle, AlertTriangle, Info, Leaf, TrendingUp, Shield, Lightbulb } from 'lucide-react'
import { PredictionResult } from '../types'

interface ResultsDisplayProps {
  result: PredictionResult
}

export default function ResultsDisplay({ result }: ResultsDisplayProps) {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'High':
        return 'text-red-600 dark:text-red-400'
      case 'Medium':
        return 'text-orange-600 dark:text-orange-400'
      case 'Low':
        return 'text-yellow-600 dark:text-yellow-400'
      case 'Very Low':
        return 'text-blue-600 dark:text-blue-400'
      default:
        return 'text-soil-600 dark:text-soil-400'
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'High':
        return <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
      case 'Medium':
        return <AlertTriangle className="w-6 h-6 text-orange-600 dark:text-orange-400" />
      case 'Low':
        return <Info className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
      case 'Very Low':
        return <CheckCircle className="w-6 h-6 text-blue-600 dark:text-blue-400" />
      default:
        return <Info className="w-6 h-6 text-soil-600 dark:text-soil-400" />
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600 dark:text-green-400'
    if (confidence >= 70) return 'text-blue-600 dark:text-blue-400'
    if (confidence >= 50) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-red-600 dark:text-red-400'
  }

  return (
    <div className="space-y-8">
      {/* Main Result Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 overflow-hidden"
      >
        <div className="p-8">
          <div className="flex items-start justify-between mb-6">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-4">
                {getSeverityIcon(result.severity)}
                <div>
                  <h3 className="text-2xl font-bold text-soil-900 dark:text-white">
                    {result.formatted_class_name}
                  </h3>
                  <p className={`text-lg font-medium ${getSeverityColor(result.severity)}`}>
                    {result.severity} Confidence
                  </p>
                </div>
              </div>
              
              <p className="text-soil-600 dark:text-soil-400 text-lg leading-relaxed">
                {result.recommendations.description}
              </p>
            </div>
            
            <div className="text-right">
              <div className="space-y-3">
                {/* Confidence Score */}
                <div className="text-center">
                  <span className={`text-3xl font-bold ${getConfidenceColor(result.confidence)}`}>
                    {result.confidence}%
                  </span>
                  <p className="text-sm text-soil-500 dark:text-soil-400">Confidence</p>
                </div>
                
                {/* Animated Confidence Bar */}
                <div className="w-32">
                  <div className="relative h-3 bg-soil-200 dark:bg-soil-700 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${result.confidence}%` }}
                      transition={{ duration: 1.5, ease: "easeOut" }}
                      className={`h-full rounded-full ${
                        result.confidence >= 90 
                          ? 'bg-gradient-to-r from-green-400 to-green-600' 
                          : result.confidence >= 70 
                          ? 'bg-gradient-to-r from-blue-400 to-blue-600'
                          : result.confidence >= 50
                          ? 'bg-gradient-to-r from-yellow-400 to-yellow-600'
                          : 'bg-gradient-to-r from-red-400 to-red-600'
                      }`}
                    />
                  </div>
                  
                  {/* Confidence Level Indicator */}
                  <div className="flex justify-between text-xs text-soil-500 dark:text-soil-400 mt-1">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                  </div>
                </div>
                
                {/* Confidence Label */}
                <div className="text-center">
                  <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${
                    result.confidence >= 90 
                      ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                      : result.confidence >= 70 
                      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400'
                      : result.confidence >= 50
                      ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
                      : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
                  }`}>
                    {result.confidence >= 90 ? 'Excellent' : 
                     result.confidence >= 70 ? 'Good' : 
                     result.confidence >= 50 ? 'Fair' : 'Poor'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Top Predictions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6"
      >
        <h4 className="text-xl font-semibold text-soil-900 dark:text-white mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-plant-600 dark:text-plant-400" />
          Top Predictions
        </h4>
        
        <div className="space-y-3">
          {result.top_predictions.map((prediction, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`p-4 rounded-lg ${
                index === 0
                  ? 'bg-plant-50 dark:bg-plant-950/20 border border-plant-200 dark:border-plant-800'
                  : 'bg-soil-50 dark:bg-soil-800/50'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    index === 0
                      ? 'bg-plant-600 text-white'
                      : 'bg-soil-300 dark:bg-soil-600 text-soil-700 dark:text-soil-300'
                  }`}>
                    {index + 1}
                  </span>
                  <span className="font-medium text-soil-900 dark:text-white">
                    {prediction.class.replace(/_/g, ' ').replace(/\s+/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                </div>
                <span className="text-lg font-semibold text-soil-700 dark:text-soil-300">
                  {prediction.confidence}%
                </span>
              </div>
              
              {/* Animated Confidence Bar for Each Prediction */}
              <div className="ml-11">
                <div className="relative h-2 bg-soil-200 dark:bg-soil-700 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${prediction.confidence}%` }}
                    transition={{ duration: 1, delay: 0.5 + index * 0.1, ease: "easeOut" }}
                    className={`h-full rounded-full ${
                      prediction.confidence >= 90 
                        ? 'bg-gradient-to-r from-green-400 to-green-600' 
                        : prediction.confidence >= 70 
                        ? 'bg-gradient-to-r from-blue-400 to-blue-600'
                        : prediction.confidence >= 50
                        ? 'bg-gradient-to-r from-yellow-400 to-yellow-600'
                        : 'bg-gradient-to-r from-red-400 to-red-600'
                    }`}
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Recommendations */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="grid md:grid-cols-2 gap-6"
      >
        {/* Treatment */}
        <div className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-lg">
              <Shield className="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
            <h4 className="text-xl font-semibold text-soil-900 dark:text-white">
              Treatment
            </h4>
          </div>
          <p className="text-soil-600 dark:text-soil-400 leading-relaxed">
            {result.recommendations.treatment}
          </p>
        </div>

        {/* Prevention */}
        <div className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-green-100 dark:bg-green-900/20 rounded-lg">
              <Lightbulb className="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
            <h4 className="text-xl font-semibold text-soil-900 dark:text-white">
              Prevention
            </h4>
          </div>
          <p className="text-soil-600 dark:text-soil-400 leading-relaxed">
            {result.recommendations.prevention}
          </p>
        </div>
      </motion.div>

      {/* Additional Info */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
        className="bg-gradient-to-r from-plant-50 to-leaf-50 dark:from-plant-950/20 dark:to-leaf-950/20 rounded-2xl border border-plant-200 dark:border-plant-800 p-6"
      >
        <div className="flex items-center gap-3 mb-4">
          <Leaf className="w-6 h-6 text-plant-600 dark:text-plant-400" />
          <h4 className="text-xl font-semibold text-soil-900 dark:text-white">
            Analysis Details
          </h4>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4 text-sm text-soil-600 dark:text-soil-400">
          <div>
            <span className="font-medium">Analysis Time:</span> {new Date(result.timestamp).toLocaleString()}
          </div>
          <div>
            <span className="font-medium">Model Confidence:</span> {result.confidence}%
          </div>
          <div>
            <span className="font-medium">Severity Level:</span> {result.severity}
          </div>
          <div>
            <span className="font-medium">Plant Type:</span> {result.predicted_class.includes('Tomato') ? 'Tomato' : result.predicted_class.includes('Potato') ? 'Potato' : result.predicted_class.includes('Pepper') ? 'Bell Pepper' : 'Unknown'}
          </div>
        </div>
      </motion.div>
    </div>
  )
}
