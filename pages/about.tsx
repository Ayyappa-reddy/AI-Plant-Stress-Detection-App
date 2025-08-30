import { motion } from 'framer-motion'
import { 
  Brain, 
  Database, 
  Target, 
  Zap, 
  Shield, 
  TrendingUp,
  Cpu,
  BarChart3,
  Leaf,
  Beaker,
  Globe,
  Users
} from 'lucide-react'

export default function AboutPage() {
  const modelStats = [
    { label: 'Accuracy', value: '93.73%', icon: Target, color: 'text-green-600' },
    { label: 'Classes', value: '15', icon: Database, color: 'text-blue-600' },
    { label: 'Response Time', value: '< 500ms', icon: Zap, color: 'text-purple-600' },
    { label: 'Training Images', value: '15,000+', icon: BarChart3, color: 'text-orange-600' }
  ]

  const capabilities = [
    {
      title: 'Disease Detection',
      description: 'Identify 15 different plant diseases and conditions with high accuracy',
      icon: Shield,
      color: 'from-red-500 to-red-600'
    },
    {
      title: 'Plant Types',
      description: 'Specialized in Solanaceae family: Tomatoes, Potatoes, Bell Peppers',
      icon: Leaf,
      color: 'from-green-500 to-green-600'
    },
    {
      title: 'Real-time Analysis',
      description: 'Instant results with detailed confidence scores and recommendations',
      icon: Zap,
      color: 'from-blue-500 to-blue-600'
    },
    {
      title: 'Treatment Guidance',
      description: 'AI-powered recommendations for disease treatment and prevention',
      icon: Beaker,
      color: 'from-purple-500 to-purple-600'
    }
  ]

  const technologyStack = [
    { name: 'PyTorch', description: 'Deep Learning Framework', icon: Brain },
    { name: 'MobileNetV2', description: 'Transfer Learning Model', icon: Cpu },
    { name: 'FastAPI', description: 'Backend API Framework', icon: Zap },
    { name: 'Next.js', description: 'Frontend Framework', icon: Globe }
  ]

  const diseaseClasses = [
    'Pepper Bacterial Spot', 'Pepper Healthy',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
    'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Spider Mites',
    'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus',
    'Tomato Healthy'
  ]

  return (
    <>
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
              <a href="/#contact" className="text-soil-700 dark:text-soil-300 hover:text-plant-600 dark:hover:text-plant-400 transition-colors">Contact</a>
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
            <div className="inline-block mb-6">
              <Brain className="w-20 h-20 text-plant-600 dark:text-plant-400 mx-auto" />
            </div>
            <h1 className="text-5xl md:text-7xl font-bold text-soil-900 dark:text-white mb-6">
              About Our <span className="text-plant-600 dark:text-plant-400">AI Technology</span>
            </h1>
            <p className="text-xl md:text-2xl text-soil-600 dark:text-soil-300 mb-8 max-w-4xl mx-auto">
              Discover how our advanced AI model revolutionizes plant disease detection with cutting-edge deep learning technology
            </p>
          </motion.div>
        </div>
      </section>

      <div className="container mx-auto px-4 pb-20">
        {/* Model Statistics */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Model Performance Metrics
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {modelStats.map((stat, index) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6 text-center"
                >
                  <div className={`inline-block p-3 bg-plant-100 dark:bg-plant-900/20 rounded-full mb-4`}>
                    <stat.icon className={`w-8 h-8 ${stat.color}`} />
                  </div>
                  <div className={`text-3xl font-bold ${stat.color} mb-2`}>
                    {stat.value}
                  </div>
                  <div className="text-soil-600 dark:text-soil-400 font-medium">
                    {stat.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Technology Overview */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Technology Stack
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {technologyStack.map((tech, index) => (
                <motion.div
                  key={tech.name}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6 text-center"
                >
                  <div className="inline-block p-3 bg-plant-100 dark:bg-plant-900/20 rounded-full mb-4">
                    <tech.icon className="w-8 h-8 text-plant-600 dark:text-plant-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-2">
                    {tech.name}
                  </h3>
                  <p className="text-soil-600 dark:text-soil-400 text-sm">
                    {tech.description}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* AI Capabilities */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              What Our AI Can Detect
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {capabilities.map((capability, index) => (
                <motion.div
                  key={capability.title}
                  initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6"
                >
                  <div className="flex items-start gap-4">
                    <div className={`p-3 bg-gradient-to-r ${capability.color} rounded-lg`}>
                      <capability.icon className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-2">
                        {capability.title}
                      </h3>
                      <p className="text-soil-600 dark:text-soil-400">
                        {capability.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Disease Classes */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Supported Disease Classes
            </h2>
            <div className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-8">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {diseaseClasses.map((disease, index) => (
                  <motion.div
                    key={disease}
                    initial={{ opacity: 0, scale: 0.9 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    viewport={{ once: true }}
                    className="flex items-center gap-3 p-3 bg-soil-50 dark:bg-soil-800/50 rounded-lg"
                  >
                    <div className="w-2 h-2 bg-plant-500 rounded-full"></div>
                    <span className="text-soil-700 dark:text-soil-300 font-medium">
                      {disease}
                    </span>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </section>

        {/* How It Works */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              How Our AI Works
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="inline-block p-4 bg-plant-100 dark:bg-plant-900/20 rounded-full mb-4">
                  <Target className="w-12 h-12 text-plant-600 dark:text-plant-400" />
                </div>
                <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-3">
                  Image Input
                </h3>
                <p className="text-soil-600 dark:text-soil-400">
                  Upload a clear photo of your plant leaf. Our system automatically preprocesses and optimizes the image for analysis.
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="inline-block p-4 bg-plant-100 dark:bg-plant-900/20 rounded-full mb-4">
                  <Brain className="w-12 h-12 text-plant-600 dark:text-plant-400" />
                </div>
                <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-3">
                  AI Analysis
                </h3>
                <p className="text-soil-600 dark:text-soil-400">
                  Our MobileNetV2 model analyzes the image using transfer learning, comparing it against thousands of training examples.
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="inline-block p-4 bg-plant-100 dark:bg-plant-900/20 rounded-full mb-4">
                  <TrendingUp className="w-12 h-12 text-plant-600 dark:text-plant-400" />
                </div>
                <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-3">
                  Results & Recommendations
                </h3>
                <p className="text-soil-600 dark:text-soil-400">
                  Get instant disease identification with confidence scores and personalized treatment recommendations.
                </p>
              </motion.div>
            </div>
          </motion.div>
        </section>

        {/* Training Data */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Training & Data
            </h2>
            <div className="bg-gradient-to-r from-plant-50 to-leaf-50 dark:from-plant-950/20 dark:to-leaf-950/20 rounded-2xl border border-plant-200 dark:border-plant-800 p-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-2xl font-semibold text-soil-900 dark:text-white mb-4">
                    Dataset Information
                  </h3>
                  <div className="space-y-3 text-soil-600 dark:text-soil-400">
                    <div className="flex items-center gap-2">
                      <Database className="w-5 h-5 text-plant-500" />
                      <span><strong>Source:</strong> PlantVillage Dataset</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-plant-500" />
                      <span><strong>Total Images:</strong> 15,000+ high-quality leaf images</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Target className="w-5 h-5 text-plant-500" />
                      <span><strong>Classes:</strong> 15 disease categories + healthy plants</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Leaf className="w-5 h-5 text-plant-500" />
                      <span><strong>Plant Types:</strong> Tomatoes, Potatoes, Bell Peppers</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h3 className="text-2xl font-semibold text-soil-900 dark:text-white mb-4">
                    Model Training
                  </h3>
                  <div className="space-y-3 text-soil-600 dark:text-soil-400">
                    <div className="flex items-center gap-2">
                      <Brain className="w-5 h-5 text-plant-500" />
                      <span><strong>Architecture:</strong> MobileNetV2 with Transfer Learning</span>
                    </div>
                                         <div className="flex items-center gap-2">
                       <TrendingUp className="w-5 h-5 text-plant-500" />
                       <span><strong>Accuracy:</strong> 93.73% on validation set</span>
                     </div>
                    <div className="flex items-center gap-2">
                      <Zap className="w-5 h-5 text-plant-500" />
                      <span><strong>Optimization:</strong> Adam optimizer with learning rate scheduling</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Shield className="w-5 h-5 text-plant-500" />
                      <span><strong>Regularization:</strong> Dropout and data augmentation</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Call to Action */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center"
          >
            <div className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-8">
              <h3 className="text-2xl font-bold text-soil-900 dark:text-white mb-4">
                Ready to Experience AI-Powered Plant Care?
              </h3>
              <p className="text-soil-600 dark:text-soil-400 mb-6 max-w-2xl mx-auto">
                Upload your first plant image and discover how our advanced AI technology can help you maintain healthy, thriving plants.
              </p>
              <a
                href="/"
                className="inline-flex items-center gap-2 px-8 py-4 bg-plant-600 hover:bg-plant-700 text-white font-semibold rounded-lg transition-colors"
              >
                <Leaf className="w-5 h-5" />
                Start Detecting Diseases
              </a>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
    </>
  )
}
