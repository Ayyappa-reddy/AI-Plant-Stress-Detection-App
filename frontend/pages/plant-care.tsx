import { motion } from 'framer-motion'
import { 
  Calendar, 
  Droplets, 
  Sun, 
  Thermometer, 
  Wind, 
  Leaf,
  Sprout,
  Shield,
  Clock,
  BarChart3,
  Zap,
  CheckCircle,
  AlertTriangle
} from 'lucide-react'

export default function PlantCarePage() {
  const seasonalCare = [
    {
      season: 'Spring',
      months: 'March - May',
      tasks: [
        'Prepare soil with organic matter',
        'Start seeds indoors for early crops',
        'Apply balanced fertilizer (10-10-10)',
        'Monitor for early pest activity',
        'Begin regular watering schedule'
      ],
      icon: Sprout,
      color: 'from-green-500 to-green-600'
    },
    {
      season: 'Summer',
      months: 'June - August',
      tasks: [
        'Maintain consistent watering (daily)',
        'Apply mulch to retain moisture',
        'Monitor for disease development',
        'Harvest regularly to encourage growth',
        'Provide shade for heat-sensitive plants'
      ],
      icon: Sun,
      color: 'from-yellow-500 to-orange-600'
    },
    {
      season: 'Fall',
      months: 'September - November',
      tasks: [
        'Reduce watering frequency',
        'Apply potassium-rich fertilizer',
        'Clean up plant debris',
        'Plant cover crops for soil health',
        'Prepare for winter protection'
      ],
      icon: Leaf,
      color: 'from-orange-500 to-red-600'
    },
    {
      season: 'Winter',
      months: 'December - February',
      tasks: [
        'Minimal watering (only when soil is dry)',
        'Protect plants from frost',
        'Plan next season\'s garden',
        'Maintain soil structure',
        'Monitor for winter pests'
      ],
      icon: Thermometer,
      color: 'from-blue-500 to-blue-600'
    }
  ]

  const soilHealth = [
    {
      aspect: 'pH Levels',
      ideal: '6.0 - 7.0 (slightly acidic to neutral)',
      description: 'Most vegetables prefer slightly acidic soil. Test pH annually and adjust with lime (raise) or sulfur (lower).',
      icon: BarChart3,
      color: 'from-purple-500 to-purple-600'
    },
    {
      aspect: 'Organic Matter',
      ideal: '3-5% of soil composition',
      description: 'Improves water retention, aeration, and nutrient availability. Add compost, manure, or leaf mold regularly.',
      icon: Leaf,
      color: 'from-green-500 to-green-600'
    },
    {
      aspect: 'Drainage',
      ideal: 'Well-draining, not waterlogged',
      description: 'Test by digging a hole and filling with water. Should drain within 24 hours. Add sand or organic matter if needed.',
      icon: Droplets,
      color: 'from-blue-500 to-blue-600'
    },
    {
      aspect: 'Nutrient Balance',
      ideal: 'NPK ratio varies by plant type',
      description: 'Nitrogen (N) for leaves, Phosphorus (P) for roots, Potassium (K) for overall health. Test soil every 2-3 years.',
      icon: Zap,
      color: 'from-yellow-500 to-orange-600'
    }
  ]

  const wateringGuide = [
    {
      plantType: 'Tomatoes',
      frequency: 'Daily during growing season',
      method: 'Deep watering at base',
      tips: [
        'Water in morning to prevent fungal diseases',
        'Avoid wetting leaves',
        'Increase frequency during fruit development',
        'Reduce in cooler weather'
      ],
      signs: {
        overwatered: 'Yellow leaves, root rot',
        underwatered: 'Wilting, blossom end rot'
      }
    },
    {
      plantType: 'Potatoes',
      frequency: '2-3 times per week',
      method: 'Even soil moisture',
      tips: [
        'Keep soil consistently moist',
        'Water deeply to encourage root growth',
        'Reduce watering as tubers mature',
        'Avoid waterlogging'
      ],
      signs: {
        overwatered: 'Soft tubers, disease',
        underwatered: 'Small tubers, cracking'
      }
    },
    {
      plantType: 'Bell Peppers',
      frequency: 'Every 2-3 days',
      method: 'Moderate, consistent watering',
      tips: [
        'Water at soil level',
        'Maintain even moisture',
        'Increase during fruit development',
        'Allow slight drying between watering'
      ],
      signs: {
        overwatered: 'Yellow leaves, poor fruit set',
        underwatered: 'Wilting, small fruits'
      }
    }
  ]

  const fertilizerGuide = [
    {
      type: 'Organic Fertilizers',
      examples: 'Compost, manure, bone meal, fish emulsion',
      benefits: [
        'Slow-release nutrients',
        'Improves soil structure',
        'Environmentally friendly',
        'Builds long-term soil health'
      ],
      application: 'Apply 2-4 weeks before planting and as side dressing',
      icon: Leaf,
      color: 'from-green-500 to-green-600'
    },
    {
      type: 'Synthetic Fertilizers',
      examples: '10-10-10, 20-20-20, specialized formulas',
      benefits: [
        'Immediate nutrient availability',
        'Precise nutrient ratios',
        'Easy to apply',
        'Quick plant response'
      ],
      application: 'Apply according to package instructions, usually every 2-4 weeks',
      icon: Zap,
      color: 'from-blue-500 to-blue-600'
    },
    {
      type: 'Foliar Sprays',
      examples: 'Liquid seaweed, fish emulsion, micronutrient solutions',
      benefits: [
        'Rapid nutrient absorption',
        'Targets specific deficiencies',
        'Quick plant response',
        'Efficient use of nutrients'
      ],
      application: 'Spray early morning or evening, avoid hot sun',
      icon: Droplets,
      color: 'from-purple-500 to-purple-600'
    }
  ]

  const pestPrevention = [
    {
      strategy: 'Cultural Control',
      methods: [
        'Crop rotation (change plant families each year)',
        'Proper spacing for air circulation',
        'Remove plant debris and weeds',
        'Use disease-resistant varieties',
        'Time planting to avoid peak pest seasons'
      ],
      icon: Shield,
      color: 'from-green-500 to-green-600'
    },
    {
      strategy: 'Biological Control',
      methods: [
        'Introduce beneficial insects (ladybugs, lacewings)',
        'Plant companion crops (marigolds, basil)',
        'Use nematodes for soil pest control',
        'Attract birds and bats to garden',
        'Maintain diverse plant ecosystem'
      ],
      icon: Sprout,
      color: 'from-blue-500 to-blue-600'
    },
    {
      strategy: 'Physical Control',
      methods: [
        'Hand-pick large pests',
        'Use row covers for protection',
        'Install barriers and traps',
        'Prune infected plant parts',
        'Use water sprays to dislodge pests'
      ],
      icon: Wind,
      color: 'from-orange-500 to-orange-600'
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
              <Leaf className="w-20 h-20 text-plant-600 dark:text-plant-400 mx-auto" />
            </div>
            <h1 className="text-5xl md:text-7xl font-bold text-soil-900 dark:text-white mb-6">
              Plant Care <span className="text-plant-600 dark:text-plant-400">Knowledge Base</span>
            </h1>
            <p className="text-xl md:text-2xl text-soil-600 dark:text-soil-300 mb-8 max-w-4xl mx-auto">
              Comprehensive guides for seasonal care, soil health, watering, and fertilization to ensure your plants thrive year-round
            </p>
          </motion.div>
        </div>
      </section>

      <div className="container mx-auto px-4 pb-20">
        {/* Seasonal Care Guide */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Seasonal Care Guide
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {seasonalCare.map((season, index) => (
                <motion.div
                  key={season.season}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 overflow-hidden"
                >
                  <div className={`bg-gradient-to-r ${season.color} p-6 text-white`}>
                    <season.icon className="w-12 h-12 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-center">{season.season}</h3>
                    <p className="text-center text-sm opacity-90">{season.months}</p>
                  </div>
                  <div className="p-6">
                    <ul className="space-y-2">
                      {season.tasks.map((task, taskIndex) => (
                        <li key={taskIndex} className="flex items-start gap-2">
                          <CheckCircle className="w-4 h-4 text-plant-500 mt-0.5 flex-shrink-0" />
                          <span className="text-sm text-soil-700 dark:text-soil-300">{task}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Soil Health */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Soil Health & Management
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {soilHealth.map((aspect, index) => (
                <motion.div
                  key={aspect.aspect}
                  initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6"
                >
                  <div className="flex items-start gap-4">
                    <div className={`p-3 bg-gradient-to-r ${aspect.color} rounded-lg`}>
                      <aspect.icon className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-soil-900 dark:text-white mb-2">
                        {aspect.aspect}
                      </h3>
                      <p className="text-sm font-medium text-plant-600 dark:text-plant-400 mb-2">
                        Ideal: {aspect.ideal}
                      </p>
                      <p className="text-soil-600 dark:text-soil-400 text-sm">
                        {aspect.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Watering Guide */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Watering Guide by Plant Type
            </h2>
            <div className="space-y-6">
              {wateringGuide.map((plant, index) => (
                <motion.div
                  key={plant.plantType}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6"
                >
                  <h3 className="text-2xl font-semibold text-soil-900 dark:text-white mb-4">
                    {plant.plantType}
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div>
                      <h4 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                        <Clock className="w-5 h-5 text-blue-500" />
                        Frequency
                      </h4>
                      <p className="text-soil-600 dark:text-soil-400">
                        {plant.frequency}
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                        <Droplets className="w-5 h-5 text-blue-500" />
                        Method
                      </h4>
                      <p className="text-soil-600 dark:text-soil-400">
                        {plant.method}
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                        <CheckCircle className="w-5 h-5 text-green-500" />
                        Best Practices
                      </h4>
                      <ul className="space-y-1">
                        {plant.tips.map((tip, tipIndex) => (
                          <li key={tipIndex} className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-plant-500 rounded-full mt-2 flex-shrink-0"></div>
                            <span className="text-sm text-soil-700 dark:text-soil-300">{tip}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <h4 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-orange-500" />
                        Warning Signs
                      </h4>
                      <div className="space-y-2">
                        <div>
                          <span className="text-sm font-medium text-red-600 dark:text-red-400">Overwatered:</span>
                          <p className="text-xs text-soil-600 dark:text-soil-400">{plant.signs.overwatered}</p>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-orange-600 dark:text-orange-400">Underwatered:</span>
                          <p className="text-xs text-soil-600 dark:text-soil-400">{plant.signs.underwatered}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Fertilizer Guide */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Fertilizer Types & Usage
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {fertilizerGuide.map((fertilizer, index) => (
                <motion.div
                  key={fertilizer.type}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 overflow-hidden"
                >
                  <div className={`bg-gradient-to-r ${fertilizer.color} p-6 text-white`}>
                    <fertilizer.icon className="w-12 h-12 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-center">{fertilizer.type}</h3>
                  </div>
                  <div className="p-6">
                    <div className="mb-4">
                      <h4 className="font-semibold text-soil-900 dark:text-white mb-2">Examples:</h4>
                      <p className="text-sm text-soil-600 dark:text-soil-400">{fertilizer.examples}</p>
                    </div>
                    
                    <div className="mb-4">
                      <h4 className="font-semibold text-soil-900 dark:text-white mb-2">Benefits:</h4>
                      <ul className="space-y-1">
                        {fertilizer.benefits.map((benefit, benefitIndex) => (
                          <li key={benefitIndex} className="flex items-start gap-2">
                            <CheckCircle className="w-3 h-3 text-plant-500 mt-0.5 flex-shrink-0" />
                            <span className="text-sm text-soil-700 dark:text-soil-300">{benefit}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-soil-900 dark:text-white mb-2">Application:</h4>
                      <p className="text-sm text-soil-600 dark:text-soil-400">{fertilizer.application}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Pest Prevention */}
        <section className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center text-soil-900 dark:text-white mb-12">
              Pest Prevention Strategies
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {pestPrevention.map((strategy, index) => (
                <motion.div
                  key={strategy.strategy}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-soil-800 rounded-2xl shadow-lg border border-soil-200 dark:border-soil-700 p-6"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className={`p-3 bg-gradient-to-r ${strategy.color} rounded-lg`}>
                      <strategy.icon className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-xl font-semibold text-soil-900 dark:text-white">
                      {strategy.strategy}
                    </h3>
                  </div>
                  <ul className="space-y-2">
                    {strategy.methods.map((method, methodIndex) => (
                      <li key={methodIndex} className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-plant-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-soil-700 dark:text-soil-300">{method}</span>
                      </li>
                    ))}
                  </ul>
                </motion.div>
              ))}
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
                Ready to Apply Your Knowledge?
              </h3>
              <p className="text-soil-600 dark:text-soil-400 mb-6 max-w-2xl mx-auto">
                Use our AI system to identify plant diseases and get personalized care recommendations based on your specific situation.
              </p>
              <a
                href="/"
                className="inline-flex items-center gap-2 px-8 py-4 bg-plant-600 hover:bg-plant-700 text-white font-semibold rounded-lg transition-colors"
              >
                <Leaf className="w-5 h-5" />
                Start Plant Care Journey
              </a>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
    </>
  )
}
