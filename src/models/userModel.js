import mongoose from "mongoose";

const ModelSchema = new mongoose.Schema({
  topic: {
    type: String,
    required: [true, "Please provide a lecture topic"],
  },
  createdAt: {
    type: Date,
    default: Date.now,
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
