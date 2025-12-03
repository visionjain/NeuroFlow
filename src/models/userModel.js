import mongoose from "mongoose";

const ModelSchema = new mongoose.Schema({
  topic: {
    type: String,
    required: [true, "Please provide a project topic"],
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  algorithm: {
    type: String,
    required: [true, "Please Choose a Algorithm"],
  },
  // Project state persistence
  state: {
    // File paths
    trainFile: String,
    testFile: String,
    datasetPath: String,
    
    // Configuration
    trainColumns: [String],
    selectedTrainColumns: [String],
    selectedOutputColumn: String,
    testSplitRatio: String,
    
    // Preprocessing settings
    selectedHandlingMissingValue: String,
    removeDuplicates: Boolean,
    enableOutlierDetection: Boolean,
    outlierMethod: String,
    zScoreThreshold: Number,
    iqrLower: Number,
    iqrUpper: Number,
    winsorLower: Number,
    winsorUpper: Number,
    encodingMethod: String,
    selectedFeatureScaling: String,
    
    // Model settings
    regularizationType: String,
    alphaValue: String,
    enableCV: Boolean,
    cvFolds: String,
    
    // Graph and exploration settings
    selectedGraphs: [String],
    selectedExplorations: [String],
    selectedEffectFeatures: [String],
    
    // Results and outputs
    logs: String,
    results: String,
    generatedGraphs: [String],
    modelTrained: Boolean,
    availableModels: mongoose.Schema.Types.Mixed,
    
    // Metadata
    lastRunAt: Date,
    isCorrupted: Boolean
  }
});

const UserSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, "please provide name"],
  },
  email: {
    type: String,
    required: [true, "please provide email"],
    unique: true,
  },
  phoneNumber: {
    type: String,
    required: [true, "please provide phone number"],
  },
  password: {
    type: String,
    required: [true, "please provide password"],
  },
  role: {
    type: String,
    enum: ["user", "turf", "admin"],
    default: "user",
  },
  dateOfBirth: {
    type: String,
    default: null,
  },
  gender: {
    type: String,
    enum: ["male", "female", "others"],
    default: null,
  },
  isVerified: {
    type: Boolean,
    default: false,
  },
  forgotPasswordToken: String,
  forgotPasswordTokenExpiry: Date,
  verifyToken: String,
  verifyTokenExpiry: Date,

  // Lectures field
  projects: [ModelSchema],
});

const User = mongoose.models.users || mongoose.model("users", UserSchema);

export default User;
