import { motion } from 'framer-motion'
import { 
  AlertTriangle, 
  Shield, 
  Beaker, 
  Wind, 
  Droplets, 
  Sun,
  Clock,
  Thermometer,
  Users,
  Leaf,
  Zap,
  CheckCircle,
  XCircle,
  Info
} from 'lucide-react'

export default function ChemicalSafetyPage() {
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
              <AlertTriangle className="w-20 h-20 text-red-600 dark:text-red-400 mx-auto" />
            </div>
            <h1 className="text-5xl md:text-7xl font-bold text-soil-900 dark:text-white mb-6">
              Chemical Safety & <span className="text-red-600 dark:text-red-400">Usage Guide</span>
            </h1>
            <p className="text-xl md:text-2xl text-soil-600 dark:text-soil-300 mb-8 max-w-4xl mx-auto">
              Essential safety information for handling pesticides, fungicides, and other agricultural chemicals responsibly
            </p>
          </motion.div>
        </div>
      </section>

      <div className="container mx-auto px-4 pb-20">
        {/* Critical Safety Warning */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <div className="bg-red-50 dark:bg-red-950/20 border-4 border-red-200 dark:border-red-800 rounded-2xl p-8 text-center">
              <AlertTriangle className="w-16 h-16 text-red-600 dark:text-red-400 mx-auto mb-4" />
              <h2 className="text-3xl font-bold text-red-800 dark:text-red-200 mb-4">
                ⚠️ CRITICAL SAFETY INFORMATION
              </h2>
              <p className="text-red-700 dark:text-red-300 text-lg mb-4">
                Pesticides can be dangerous if not handled properly. Always read and follow the label instructions exactly.
                When in doubt, consult with agricultural professionals or extension services.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="bg-red-100 dark:bg-red-900/30 p-3 rounded-lg">
                  <strong>Never</strong> exceed recommended rates
                </div>
                <div className="bg-red-100 dark:bg-red-900/30 p-3 rounded-lg">
                  <strong>Always</strong> wear protective equipment
                </div>
                <div className="bg-red-100 dark:bg-red-900/30 p-3 rounded-lg">
                  <strong>Keep</strong> chemicals away from children
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Safety Equipment */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Required Safety Equipment
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-gradient-to-r from-red-500 to-red-600 rounded-lg">
                    <Shield className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-2">
                      Protective Clothing
                    </h3>
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-plant-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Long-sleeved shirt and pants</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-plant-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Chemical-resistant gloves</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-plant-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Safety goggles</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg">
                    <Beaker className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-2">
                      Application Safety
                    </h3>
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-plant-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Read labels carefully</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-plant-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Follow dosage rates</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-plant-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Clean equipment after use</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Emergency Procedures */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Emergency Procedures
            </h2>
            <div className="space-y-6">
              <div className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6">
                <h3 className="text-2xl font-semibold text-soil-900 dark:text-white mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-8 h-8 text-red-500" />
                  Chemical Contact with Skin
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-lg font-semibold text-soil-900 dark:text-white mb-3">Immediate Actions:</h4>
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Remove contaminated clothing</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Rinse with water for 15-20 minutes</span>
                      </li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold text-soil-900 dark:text-white mb-3">Medical Attention:</h4>
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Call Poison Control: 1-800-222-1222</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">Seek medical help if needed</span>
                      </li>
                    </ul>
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
                Need Help with Chemical Application?
              </h3>
              <p className="text-soil-600 dark:text-soil-400 mb-6 max-w-2xl mx-auto">
                Use our AI system to identify plant diseases and get personalized treatment recommendations, including safe chemical usage guidelines.
              </p>
              <a
                href="/"
                className="inline-flex items-center gap-2 px-8 py-4 bg-plant-600 hover:bg-plant-700 text-white font-semibold rounded-lg transition-colors"
              >
                <Shield className="w-5 h-5" />
                Get Safe Treatment Plan
              </a>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
    </>
  )
}
