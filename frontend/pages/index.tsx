import { useState } from 'react'
import Head from 'next/head'
import { motion } from 'framer-motion'
import { Leaf, Upload, Camera, Brain, History, BookOpen, Mail, Github, Linkedin } from 'lucide-react'
import ImageUpload from '../components/ImageUpload'
import ResultsDisplay from '../components/ResultsDisplay'
import HistorySection from '../components/HistorySection'
import KnowledgeBase from '../components/KnowledgeBase'
import ContactForm from '../components/ContactForm'
import { PredictionResult } from '../types'

export default function Home() {
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handlePredictionComplete = (result: PredictionResult) => {
    setPredictionResult(result)
    setIsAnalyzing(false)
  }

  const handleAnalysisStart = () => {
    setIsAnalyzing(true)
    setPredictionResult(null)
  }

  return (
    <>
      <Head>
        <title>PlantGuard AI - AI Plant Disease Detection</title>
        <meta name="description" content="Advanced AI-powered plant disease detection using deep learning. Get instant diagnosis and treatment recommendations for your plants." />
        <meta name="keywords" content="plant disease detection, AI plant health, leaf analysis, plant stress detection, agricultural AI, crop protection, plant diagnosis, deep learning, machine learning" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* Navigation Header */}
      <nav className="bg-white/80 dark:bg-soil-900/80 backdrop-blur-md border-b border-soil-200 dark:border-soil-700 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Leaf className="w-8 h-8 text-plant-600 dark:text-plant-400" />
              <span className="text-xl font-bold text-soil-900 dark:text-white">PlantGuard AI</span>
            </div>
            <div className="hidden md:flex items-center gap-6">
              <a href="/" className="text-soil-700 dark:text-soil-300 hover:text-plant-600 dark:hover:text-plant-400 transition-colors">Home</a>
              <a href="/about" className="text-soil-700 dark:text-soil-300 hover:text-plant-600 dark:hover:text-plant-400 transition-colors">About</a>
              <a href="/treatment-guide" className="text-soil-700 dark:text-soil-300 hover:text-plant-600 dark:hover:text-plant-400 transition-colors">Treatment Guide</a>
              <a href="/plant-care" className="text-soil-700 dark:text-soil-300 hover:text-plant-600 dark:hover:text-plant-400 transition-colors">Plant Care</a>
              <a href="/chemical-safety" className="text-soil-700 dark:text-soil-300 hover:text-plant-600 dark:hover:text-plant-400 transition-colors">Chemical Safety</a>
              <a href="#contact" className="text-soil-700 dark:text-soil-300 hover:text-plant-600 dark:hover:text-plant-400 transition-colors">Contact</a>
            </div>
          </div>
        </div>
      </nav>

      <div className="min-h-screen bg-gradient-to-br from-plant-50 via-white to-leaf-50 dark:from-soil-950 dark:via-soil-900 dark:to-soil-800">
        {/* Hero Section */}
        <section className="relative overflow-hidden">
          <div className="absolute inset-0 bg-leaf-pattern opacity-5"></div>
          <div className="relative container mx-auto px-4 py-20 text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                className="inline-block mb-6"
              >
                <Leaf className="w-20 h-20 text-plant-600 dark:text-plant-400 mx-auto" />
              </motion.div>
              
              <h1 className="text-5xl md:text-7xl font-bold text-soil-900 dark:text-white mb-6">
                PlantGuard
                <span className="block text-plant-600 dark:text-plant-400">AI</span>
              </h1>
              
              <p className="text-xl md:text-2xl text-soil-600 dark:text-soil-300 mb-8 max-w-3xl mx-auto">
                Upload a photo of your plant leaf and get instant AI-powered disease detection. 
                Identify plant stress, diseases, and get personalized treatment recommendations.
              </p>
              
              <div className="flex flex-wrap justify-center gap-4 text-soil-600 dark:text-soil-400">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-plant-500" />
                  <span>AI-Powered Analysis</span>
                </div>
                <div className="flex items-center gap-2">
                  <Camera className="w-5 h-5 text-plant-500" />
                  <span>Instant Results</span>
                </div>
                <div className="flex items-center gap-2">
                  <Leaf className="w-5 h-5 text-plant-500" />
                  <span>Expert Recommendations</span>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Main Content */}
        <div className="container mx-auto px-4 pb-20">
          {/* Upload Section */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mb-20"
          >
            <div className="max-w-4xl mx-auto">
              <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-8">
                Analyze Your Plant
              </h2>
              <ImageUpload 
                onAnalysisStart={handleAnalysisStart}
                onPredictionComplete={handlePredictionComplete}
                isAnalyzing={isAnalyzing}
              />
            </div>
          </motion.section>

          {/* Results Section */}
          {predictionResult && (
            <motion.section
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="mb-20"
            >
              <div className="max-w-4xl mx-auto">
                <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-8">
                  Analysis Results
                </h2>
                <ResultsDisplay result={predictionResult} />
              </div>
            </motion.section>
          )}

          {/* History Section */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="mb-20"
          >
            <div className="max-w-6xl mx-auto">
              <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-8">
                <History className="inline-block w-8 h-8 mr-3 text-plant-600 dark:text-plant-400" />
                Scan History
              </h2>
              <HistorySection />
            </div>
          </motion.section>

          {/* Knowledge Base Section */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="mb-20"
          >
            <div className="max-w-6xl mx-auto">
              <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-8">
                <BookOpen className="inline-block w-8 h-8 mr-3 text-plant-600 dark:text-plant-400" />
                Plant Disease Knowledge Base
              </h2>
              <KnowledgeBase />
            </div>
          </motion.section>

          {/* Contact Section */}
          <motion.section
            id="contact"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="mb-20"
          >
            <div className="max-w-4xl mx-auto">
              <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-8">
                <Mail className="inline-block w-8 h-8 mr-3 text-plant-600 dark:text-plant-400" />
                Get in Touch
              </h2>
              <ContactForm />
            </div>
          </motion.section>
        </div>

        {/* Footer */}
        <footer className="bg-soil-900 dark:bg-soil-950 text-white py-12">
          <div className="container mx-auto px-4 text-center">
            <div className="flex justify-center items-center mb-6">
              <Leaf className="w-8 h-8 text-plant-400 mr-3" />
              <span className="text-xl font-semibold">PlantGuard AI</span>
            </div>
            
            <p className="text-soil-300 mb-6 max-w-2xl mx-auto">
              Advanced AI-powered plant disease detection using deep learning. Get instant diagnosis 
              and treatment recommendations for your plants. Built with cutting-edge technology 
              to help farmers and gardeners protect their crops.
            </p>
            
            <div className="flex justify-center gap-6 mb-6">
              <a 
                href="https://github.com/YOUR_USERNAME" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-soil-300 hover:text-plant-400 transition-colors"
              >
                <Github className="w-6 h-6" />
              </a>
              <a 
                href="https://linkedin.com/in/YOUR_USERNAME" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-soil-300 hover:text-plant-400 transition-colors"
              >
                <Linkedin className="w-6 h-6" />
              </a>
            </div>
            
            <div className="border-t border-soil-700 pt-6">
              <p className="text-soil-400 text-sm">
                © 2024 PlantGuard AI. Built with Next.js, FastAPI, and PyTorch.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  )
}
