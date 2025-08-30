import { motion } from 'framer-motion'
import { 
  Shield, 
  Leaf, 
  Droplets, 
  Sun, 
  AlertTriangle, 
  CheckCircle,
  Beaker,
  Sprout,
  Clock,
  Thermometer,
  Wind,
  Zap
} from 'lucide-react'

export default function TreatmentGuidePage() {
  const treatmentMethods = [
    {
      title: 'Chemical Treatments',
      description: 'Professional pesticides and fungicides for severe infections',
      icon: Beaker,
      color: 'from-red-500 to-red-600',
      items: [
        'Copper-based fungicides for bacterial diseases',
        'Sulfur-based treatments for fungal infections',
        'Systemic insecticides for pest control',
        'Contact pesticides for immediate relief'
      ]
    },
    {
      title: 'Natural Remedies',
      description: 'Organic solutions using household ingredients',
      icon: Sprout,
      color: 'from-green-500 to-green-600',
      items: [
        'Neem oil for pest control',
        'Baking soda solution for fungal issues',
        'Garlic and chili pepper sprays',
        'Beneficial insect introduction'
      ]
    },
    {
      title: 'Cultural Practices',
      description: 'Prevention through proper plant care',
      icon: Leaf,
      color: 'from-blue-500 to-blue-600',
      items: [
        'Proper spacing between plants',
        'Crop rotation strategies',
        'Sanitation and debris removal',
        'Optimal watering practices'
      ]
    }
  ]

  const safetyGuidelines = [
    {
      title: 'Personal Protection',
      icon: Shield,
      items: [
        'Wear long sleeves, pants, and closed-toe shoes',
        'Use chemical-resistant gloves (nitrile or neoprene)',
        'Wear safety goggles or face shield',
        'Use respiratory protection for dust applications',
        'Avoid contact with skin and eyes'
      ]
    },
    {
      title: 'Environmental Safety',
      icon: Leaf,
      items: [
        'Apply during calm weather (wind < 10 mph)',
        'Avoid application before rain (within 24 hours)',
        'Keep chemicals away from water sources',
        'Follow buffer zone requirements',
        'Store chemicals in secure, dry locations'
      ]
    },
    {
      title: 'Application Safety',
      icon: Beaker,
      items: [
        'Read and follow label instructions exactly',
        'Never exceed recommended dosage rates',
        'Mix chemicals in well-ventilated areas',
        'Clean equipment thoroughly after use',
        'Dispose of empty containers properly'
      ]
    }
  ]

  const diseaseSpecificTreatments = [
    {
      disease: 'Bacterial Spot',
      symptoms: 'Dark, water-soaked lesions on leaves and fruits',
      treatments: [
        'Remove and destroy infected plant parts',
        'Apply copper-based bactericides',
        'Improve air circulation around plants',
        'Avoid overhead watering'
      ],
      prevention: [
        'Use disease-resistant varieties',
        'Practice crop rotation',
        'Maintain proper plant spacing',
        'Control weeds and debris'
      ]
    },
    {
      disease: 'Early Blight',
      symptoms: 'Dark brown spots with concentric rings on lower leaves',
      treatments: [
        'Remove infected leaves immediately',
        'Apply chlorothalonil or mancozeb',
        'Improve air circulation',
        'Mulch around plants'
      ],
      prevention: [
        'Avoid overhead irrigation',
        'Maintain adequate nitrogen levels',
        'Remove plant debris',
        'Use resistant varieties'
      ]
    },
    {
      disease: 'Late Blight',
      symptoms: 'Water-soaked lesions that rapidly expand',
      treatments: [
        'Remove all infected plants immediately',
        'Apply copper-based fungicides',
        'Improve drainage and air flow',
        'Consider systemic fungicides'
      ],
      prevention: [
        'Monitor weather conditions',
        'Apply preventive fungicides',
        'Ensure proper spacing',
        'Avoid working in wet conditions'
      ]
    }
  ]

  const applicationMethods = [
    {
      method: 'Foliar Spray',
      description: 'Direct application to plant leaves',
      equipment: 'Backpack sprayer, hand pump sprayer',
      tips: [
        'Apply early morning or late evening',
        'Cover both sides of leaves thoroughly',
        'Avoid spraying during peak sun hours',
        'Use fine mist for better coverage'
      ]
    },
    {
      method: 'Soil Drench',
      description: 'Application to soil around plant roots',
      equipment: 'Watering can, drip irrigation system',
      tips: [
        'Apply to moist soil, not saturated',
        'Target root zone area',
        'Avoid runoff and waste',
        'Water in thoroughly after application'
      ]
    },
    {
      method: 'Seed Treatment',
      description: 'Coating seeds before planting',
      equipment: 'Seed treater, mixing container',
      tips: [
        'Follow label rates exactly',
        'Ensure even coverage',
        'Plant treated seeds immediately',
        'Store properly if delayed'
      ]
    }
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
              <Shield className="w-20 h-20 text-plant-600 dark:text-plant-400 mx-auto" />
            </div>
            <h1 className="text-5xl md:text-7xl font-bold text-soil-900 dark:text-white mb-6">
              Treatment & <span className="text-plant-600 dark:text-plant-400">Care Guide</span>
            </h1>
            <p className="text-xl md:text-2xl text-soil-600 dark:text-soil-300 mb-8 max-w-4xl mx-auto">
              Comprehensive guide to treating plant diseases safely and effectively using chemical and natural methods
            </p>
          </motion.div>
        </div>
      </section>

      <div className="container mx-auto px-4 pb-20">
        {/* Treatment Methods Overview */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Treatment Approaches
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {treatmentMethods.map((method, index) => (
                <motion.div
                  key={method.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 overflow-hidden"
                >
                  <div className={`bg-gradient-to-r ${method.color} p-6 text-white`}>
                    <method.icon className="w-12 h-12 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-center">{method.title}</h3>
                  </div>
                  <div className="p-6">
                    <p className="text-soil-600 dark:text-soil-400 mb-4 text-center">
                      {method.description}
                    </p>
                    <ul className="space-y-2">
                      {method.items.map((item, itemIndex) => (
                        <li key={itemIndex} className="flex items-start gap-2">
                          <CheckCircle className="w-4 h-4 text-plant-500 mt-0.5 flex-shrink-0" />
                          <span className="text-sm text-soil-700 dark:text-soil-300">{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Safety Guidelines */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Safety Guidelines
            </h2>
            <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-2xl p-8 mb-8">
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
                <h3 className="text-2xl font-semibold text-red-800 dark:text-red-200">
                  ⚠️ Critical Safety Information
                </h3>
              </div>
              <p className="text-red-700 dark:text-red-300 text-lg">
                Always read and follow pesticide label instructions. Improper use can harm you, your plants, and the environment. 
                When in doubt, consult with agricultural professionals.
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {safetyGuidelines.map((guideline, index) => (
                <motion.div
                  key={guideline.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <guideline.icon className="w-8 h-8 text-plant-600 dark:text-plant-400" />
                    <h3 className="text-xl font-semibold text-soil-900 dark:text-white">
                      {guideline.title}
                    </h3>
                  </div>
                  <ul className="space-y-2">
                    {guideline.items.map((item, itemIndex) => (
                      <li key={itemIndex} className="flex items-start gap-2">
                        <div className="w-2 h-2 bg-plant-500 rounded-full mt-2 flex-shrink-0"></div>
                        <span className="text-sm text-soil-700 dark:text-soil-300">{item}</span>
                      </li>
                    ))}
                  </ul>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Disease-Specific Treatments */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Disease-Specific Treatment Plans
            </h2>
            <div className="space-y-6">
              {diseaseSpecificTreatments.map((disease, index) => (
                <motion.div
                  key={disease.disease}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6"
                >
                  <h3 className="text-2xl font-semibold text-soil-900 dark:text-white mb-4">
                    {disease.disease}
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <h4 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-orange-500" />
                        Symptoms
                      </h4>
                      <p className="text-soil-600 dark:text-soil-400">
                        {disease.symptoms}
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                        <Beaker className="w-5 h-5 text-blue-500" />
                        Treatments
                      </h4>
                      <ul className="space-y-2">
                        {disease.treatments.map((treatment, treatIndex) => (
                          <li key={treatIndex} className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                            <span className="text-sm text-soil-700 dark:text-soil-300">{treatment}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <h4 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                        <Shield className="w-5 h-5 text-green-500" />
                        Prevention
                      </h4>
                      <ul className="space-y-2">
                        {disease.prevention.map((prevent, preventIndex) => (
                          <li key={preventIndex} className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                            <span className="text-sm text-soil-700 dark:text-soil-300">{prevent}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Application Methods */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Application Methods
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {applicationMethods.map((method, index) => (
                <motion.div
                  key={method.method}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6"
                >
                  <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-3">
                    {method.method}
                  </h3>
                  <p className="text-soil-600 dark:text-soil-400 mb-4">
                    {method.description}
                  </p>
                  
                  <div className="mb-4">
                    <h4 className="font-semibold text-soil-900 dark:text-white mb-2">Equipment Needed:</h4>
                    <p className="text-sm text-soil-600 dark:text-soil-400">{method.equipment}</p>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-soil-900 dark:text-white mb-2">Best Practices:</h4>
                    <ul className="space-y-1">
                      {method.tips.map((tip, tipIndex) => (
                        <li key={tipIndex} className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-plant-500 rounded-full mt-2 flex-shrink-0"></div>
                          <span className="text-sm text-soil-700 dark:text-soil-300">{tip}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Environmental Considerations */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Environmental Considerations
            </h2>
            <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950/20 dark:to-blue-950/20 rounded-2xl border border-green-200 dark:border-green-800 p-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-2xl font-semibold text-soil-900 dark:text-white mb-4 flex items-center gap-2">
                    <Leaf className="w-6 h-6 text-green-500" />
                    Weather Conditions
                  </h3>
                  <div className="space-y-3 text-soil-600 dark:text-soil-400">
                    <div className="flex items-center gap-2">
                      <Thermometer className="w-5 h-5 text-green-500" />
                      <span><strong>Temperature:</strong> Apply between 60-85°F (15-29°C)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Wind className="w-5 h-5 text-green-500" />
                      <span><strong>Wind:</strong> Avoid application when wind &gt; 10 mph</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Droplets className="w-5 h-5 text-green-500" />
                      <span><strong>Rain:</strong> No rain expected for 24 hours</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Sun className="w-5 h-5 text-green-500" />
                      <span><strong>Timing:</strong> Early morning or late evening</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-2xl font-semibold text-soil-900 dark:text-white mb-4 flex items-center gap-2">
                    <Shield className="w-6 h-6 text-blue-500" />
                    Protection Measures
                  </h3>
                  <div className="space-y-3 text-soil-600 dark:text-soil-400">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-blue-500" />
                      <span>Maintain buffer zones around water sources</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-blue-500" />
                      <span>Use integrated pest management (IPM)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-blue-500" />
                      <span>Rotate chemical classes to prevent resistance</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-blue-500" />
                      <span>Monitor beneficial insect populations</span>
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
                Need Help with a Specific Disease?
              </h3>
              <p className="text-soil-600 dark:text-soil-400 mb-6 max-w-2xl mx-auto">
                Upload a photo of your plant and get personalized treatment recommendations from our AI system.
              </p>
              <a
                href="/"
                className="inline-flex items-center gap-2 px-8 py-4 bg-plant-600 hover:bg-plant-700 text-white font-semibold rounded-lg transition-colors"
              >
                <Zap className="w-5 h-5" />
                Get AI Diagnosis
              </a>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
    </>
  )
}
