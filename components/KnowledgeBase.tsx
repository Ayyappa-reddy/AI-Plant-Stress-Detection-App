import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BookOpen, AlertTriangle, Leaf, Shield, Lightbulb, Search, X } from 'lucide-react'
import { DiseaseInfo } from '../types'

const DISEASE_DATA: DiseaseInfo[] = [
  {
    name: "Pepper__bell___Bacterial_spot",
    formatted_name: "Bell Pepper Bacterial Spot",
    description: "A serious bacterial disease that can cause significant yield loss in bell pepper plants.",
    symptoms: ["Dark, water-soaked lesions on leaves", "Brown spots with yellow halos", "Lesions on stems and fruits", "Leaf drop and defoliation"],
    treatment: "Remove infected plants, apply copper-based bactericides, and practice crop rotation.",
    prevention: "Use disease-free seeds, avoid overhead irrigation, maintain proper spacing, and sanitize tools.",
    severity: "High",
    affected_plants: ["Bell Peppers", "Hot Peppers", "Other Solanaceae"]
  },
  {
    name: "Pepper__bell___healthy",
    formatted_name: "Bell Pepper Healthy",
    description: "Healthy bell pepper plants with no visible disease symptoms.",
    symptoms: ["No symptoms - plant appears healthy"],
    treatment: "No treatment needed. Continue with regular care and monitoring.",
    prevention: "Maintain good growing conditions, proper watering, and regular inspection.",
    severity: "Low",
    affected_plants: ["Bell Peppers"]
  },
  {
    name: "Potato___Early_blight",
    formatted_name: "Potato Early Blight",
    description: "A common fungal disease that affects potato leaves and stems, typically appearing early in the growing season.",
    symptoms: ["Dark brown spots with concentric rings", "Yellowing of leaves", "Premature leaf drop", "Lesions on stems"],
    treatment: "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves.",
    prevention: "Ensure proper spacing, avoid overhead irrigation, rotate crops, and remove plant debris.",
    severity: "Medium",
    affected_plants: ["Potatoes", "Tomatoes", "Other Solanaceae"]
  },
  {
    name: "Potato___Late_blight",
    formatted_name: "Potato Late Blight",
    description: "A devastating disease that can destroy entire potato crops quickly, especially in wet conditions.",
    symptoms: ["Water-soaked lesions on leaves", "White fungal growth on undersides", "Rapid leaf death", "Lesions on tubers"],
    treatment: "Apply fungicides immediately. Remove and destroy infected plants.",
    prevention: "Plant resistant varieties, avoid overhead irrigation, monitor weather conditions, and space plants properly.",
    severity: "Critical",
    affected_plants: ["Potatoes", "Tomatoes", "Other Solanaceae"]
  },
  {
    name: "Potato___healthy",
    formatted_name: "Potato Healthy",
    description: "Healthy potato plants with no visible disease symptoms.",
    symptoms: ["No symptoms - plant appears healthy"],
    treatment: "No treatment needed. Continue with regular care and monitoring.",
    prevention: "Maintain good growing conditions, proper watering, and regular inspection.",
    severity: "Low",
    affected_plants: ["Potatoes"]
  },
  {
    name: "Tomato_Bacterial_spot",
    formatted_name: "Tomato Bacterial Spot",
    description: "Bacterial spot causes dark, water-soaked lesions on tomato leaves and fruits, reducing yield and quality.",
    symptoms: ["Dark, water-soaked lesions", "Yellow halos around spots", "Lesions on fruits", "Leaf drop"],
    treatment: "Remove infected plants, apply copper-based bactericides, and practice crop rotation.",
    prevention: "Use disease-free seeds, avoid overhead irrigation, maintain proper spacing, and sanitize tools.",
    severity: "High",
    affected_plants: ["Tomatoes", "Peppers", "Other Solanaceae"]
  },
  {
    name: "Tomato_Early_blight",
    formatted_name: "Tomato Early Blight",
    description: "Early blight causes dark brown spots with concentric rings on tomato leaves, typically starting on lower leaves.",
    symptoms: ["Dark brown spots with rings", "Yellowing of leaves", "Premature leaf drop", "Lesions on stems"],
    treatment: "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves.",
    prevention: "Ensure proper spacing, avoid overhead irrigation, rotate crops, and remove plant debris.",
    severity: "Medium",
    affected_plants: ["Tomatoes", "Potatoes", "Other Solanaceae"]
  },
  {
    name: "Tomato_Late_blight",
    formatted_name: "Tomato Late Blight",
    description: "Late blight is a serious disease that can destroy tomato plants rapidly, especially in humid conditions.",
    symptoms: ["Water-soaked lesions", "White fungal growth", "Rapid leaf death", "Lesions on fruits"],
    treatment: "Apply fungicides immediately. Remove and destroy infected plants.",
    prevention: "Plant resistant varieties, avoid overhead irrigation, monitor humidity, and space plants properly.",
    severity: "Critical",
    affected_plants: ["Tomatoes", "Potatoes", "Other Solanaceae"]
  },
  {
    name: "Tomato_Leaf_Mold",
    formatted_name: "Tomato Leaf Mold",
    description: "Leaf mold is a fungal disease that thrives in humid conditions and can affect greenhouse tomatoes.",
    symptoms: ["Yellow spots on upper leaf surfaces", "Olive-green mold on undersides", "Leaf curling", "Premature leaf drop"],
    treatment: "Improve air circulation, reduce humidity, and apply fungicides if necessary.",
    prevention: "Ensure proper spacing, avoid overhead irrigation, maintain good ventilation, and control humidity.",
    severity: "Medium",
    affected_plants: ["Tomatoes", "Other Solanaceae"]
  },
  {
    name: "Tomato_Septoria_leaf_spot",
    formatted_name: "Tomato Septoria Leaf Spot",
    description: "Septoria leaf spot causes small, dark spots with gray centers on tomato leaves, leading to defoliation.",
    symptoms: ["Small, dark spots with gray centers", "Yellow halos around spots", "Leaf yellowing", "Premature leaf drop"],
    treatment: "Remove infected leaves and apply fungicides containing chlorothalonil.",
    prevention: "Avoid overhead irrigation, maintain proper spacing, rotate crops, and remove plant debris.",
    severity: "Medium",
    affected_plants: ["Tomatoes", "Other Solanaceae"]
  },
  {
    name: "Tomato_Spider_mites_Two_spotted_spider_mite",
    formatted_name: "Tomato Spider Mites",
    description: "Spider mites are tiny pests that can cause significant damage to tomato plants, especially in dry conditions.",
    symptoms: ["Fine webbing on leaves", "Yellow stippling on leaves", "Leaf curling and browning", "Reduced plant vigor"],
    treatment: "Apply insecticidal soap or neem oil. Use predatory mites for biological control.",
    prevention: "Monitor regularly, maintain proper humidity, avoid over-fertilization, and use beneficial insects.",
    severity: "Medium",
    affected_plants: ["Tomatoes", "Many other plants"]
  },
  {
    name: "Tomato__Target_Spot",
    formatted_name: "Tomato Target Spot",
    description: "Target spot causes dark brown spots with target-like rings on tomato leaves, reducing photosynthesis.",
    symptoms: ["Dark brown spots with rings", "Yellow halos around spots", "Leaf yellowing", "Premature leaf drop"],
    treatment: "Remove infected leaves and apply fungicides containing chlorothalonil.",
    prevention: "Ensure proper spacing, avoid overhead irrigation, rotate crops, and remove plant debris.",
    severity: "Medium",
    affected_plants: ["Tomatoes", "Other Solanaceae"]
  },
  {
    name: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    formatted_name: "Tomato Yellow Leaf Curl Virus",
    description: "This viral disease causes yellowing and curling of tomato leaves, transmitted by whiteflies.",
    symptoms: ["Yellowing of leaves", "Upward leaf curling", "Stunted growth", "Reduced fruit production"],
    treatment: "Remove infected plants immediately. Control whitefly populations.",
    prevention: "Use virus-resistant varieties, control whiteflies, remove weeds, and use reflective mulches.",
    severity: "High",
    affected_plants: ["Tomatoes", "Other Solanaceae"]
  },
  {
    name: "Tomato__Tomato_mosaic_virus",
    formatted_name: "Tomato Mosaic Virus",
    description: "Tomato mosaic virus causes mottled leaves and stunted growth, reducing yield and quality.",
    symptoms: ["Mottled leaves", "Stunted growth", "Leaf distortion", "Reduced fruit production"],
    treatment: "Remove infected plants immediately. Disinfect tools and hands.",
    prevention: "Use virus-free seeds, control aphids, practice good hygiene, and avoid smoking near plants.",
    severity: "High",
    affected_plants: ["Tomatoes", "Other Solanaceae"]
  },
  {
    name: "Tomato_healthy",
    formatted_name: "Tomato Healthy",
    description: "Healthy tomato plants with no visible disease symptoms.",
    symptoms: ["No symptoms - plant appears healthy"],
    treatment: "No treatment needed. Continue with regular care and monitoring.",
    prevention: "Maintain good growing conditions, proper watering, and regular inspection.",
    severity: "Low",
    affected_plants: ["Tomatoes"]
  }
]

export default function KnowledgeBase() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedDisease, setSelectedDisease] = useState<DiseaseInfo | null>(null)
  const [filteredDiseases, setFilteredDiseases] = useState<DiseaseInfo[]>(DISEASE_DATA)

  const handleSearch = (term: string) => {
    setSearchTerm(term)
    if (term.trim() === '') {
      setFilteredDiseases(DISEASE_DATA)
    } else {
      const filtered = DISEASE_DATA.filter(disease =>
        disease.formatted_name.toLowerCase().includes(term.toLowerCase()) ||
        disease.description.toLowerCase().includes(term.toLowerCase()) ||
        disease.symptoms.some(symptom => symptom.toLowerCase().includes(term.toLowerCase()))
      )
      setFilteredDiseases(filtered)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Critical':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
      case 'High':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400'
      case 'Medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
      case 'Low':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
      default:
        return 'bg-soil-100 text-soil-800 dark:bg-soil-800/50 dark:text-soil-400'
    }
  }

  return (
    <div className="space-y-6">
      {/* Search Bar */}
      <div className="relative max-w-md mx-auto">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-soil-400" />
        <input
          type="text"
          placeholder="Search diseases, symptoms, or plants..."
          value={searchTerm}
          onChange={(e) => handleSearch(e.target.value)}
          className="w-full pl-10 pr-4 py-3 border border-soil-300 dark:border-soil-600 rounded-lg bg-white dark:bg-soil-800 text-soil-900 dark:text-white placeholder-soil-400 dark:placeholder-soil-500 focus:ring-2 focus:ring-plant-500 focus:border-transparent"
        />
      </div>

      {/* Disease Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <AnimatePresence>
          {filteredDiseases.map((disease, index) => (
            <motion.div
              key={disease.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="bg-white dark:bg-soil-800 rounded-xl shadow-lg border border-soil-200 dark:border-soil-700 overflow-hidden hover:shadow-xl transition-shadow cursor-pointer"
              onClick={() => setSelectedDisease(disease)}
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="font-semibold text-soil-900 dark:text-white text-lg mb-2 leading-tight">
                      {disease.formatted_name}
                    </h3>
                    <p className="text-soil-600 dark:text-soil-400 text-sm leading-relaxed line-clamp-3">
                      {disease.description}
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getSeverityColor(disease.severity)}`}>
                      {disease.severity} Risk
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-xs text-soil-500 dark:text-soil-400">
                    <Leaf className="w-3 h-3" />
                    <span>{disease.affected_plants.join(', ')}</span>
                  </div>

                  <div className="pt-2 border-t border-soil-200 dark:border-soil-700">
                    <p className="text-xs text-soil-500 dark:text-soil-400">
                      Click to learn more about symptoms, treatment, and prevention
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Disease Detail Modal */}
      <AnimatePresence>
        {selectedDisease && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedDisease(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white dark:bg-soil-800 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-6">
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold text-soil-900 dark:text-white mb-2">
                      {selectedDisease.formatted_name}
                    </h2>
                    <span className={`inline-block px-3 py-1 text-sm font-medium rounded-full ${getSeverityColor(selectedDisease.severity)}`}>
                      {selectedDisease.severity} Risk
                    </span>
                  </div>
                  <button
                    onClick={() => setSelectedDisease(null)}
                    className="p-2 text-soil-400 hover:text-soil-600 dark:hover:text-soil-300 transition-colors"
                  >
                    <X className="w-6 h-6" />
                  </button>
                </div>

                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                      <BookOpen className="w-5 h-5 text-plant-600 dark:text-plant-400" />
                      Description
                    </h3>
                    <p className="text-soil-600 dark:text-soil-400 leading-relaxed">
                      {selectedDisease.description}
                    </p>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                      <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" />
                      Symptoms
                    </h3>
                    <ul className="space-y-2">
                      {selectedDisease.symptoms.map((symptom, index) => (
                        <li key={index} className="flex items-start gap-2 text-soil-600 dark:text-soil-400">
                          <span className="w-2 h-2 bg-red-400 rounded-full mt-2 flex-shrink-0"></span>
                          {symptom}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                        <Shield className="w-5 h-5 text-red-600 dark:text-red-400" />
                        Treatment
                      </h3>
                      <p className="text-soil-600 dark:text-soil-400 leading-relaxed">
                        {selectedDisease.treatment}
                      </p>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                        <Lightbulb className="w-5 h-5 text-green-600 dark:text-green-400" />
                        Prevention
                      </h3>
                      <p className="text-soil-600 dark:text-soil-400 leading-relaxed">
                        {selectedDisease.prevention}
                      </p>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-soil-900 dark:text-white mb-3 flex items-center gap-2">
                      <Leaf className="w-5 h-5 text-plant-600 dark:text-plant-400" />
                      Affected Plants
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedDisease.affected_plants.map((plant, index) => (
                        <span
                          key={index}
                          className="px-3 py-1 bg-plant-100 dark:bg-plant-900/20 text-plant-700 dark:text-plant-300 rounded-full text-sm"
                        >
                          {plant}
                        </span>
                      ))}
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
