export interface PredictionResult {
  predicted_class: string
  formatted_class_name: string
  confidence: number
  severity: string
  top_predictions: Array<{
    class: string
    confidence: number
    class_index: number
  }>
  recommendations: {
    description: string
    treatment: string
    prevention: string
  }
  timestamp: string
}

export interface ScanHistory {
  id: string
  image: string
  result: PredictionResult
  date: string
}

export interface DiseaseInfo {
  name: string
  formatted_name: string
  description: string
  symptoms: string[]
  treatment: string
  prevention: string
  severity: 'Low' | 'Medium' | 'High' | 'Critical'
  affected_plants: string[]
}

export interface ContactFormData {
  name: string
  email: string
  message: string
}

export interface UploadState {
  isDragging: boolean
  selectedFile: File | null
  preview: string | null
  error: string | null
}
