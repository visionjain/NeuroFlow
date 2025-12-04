"use client";

/**
 * LOGISTIC REGRESSION - COMPLETE FEATURE SET
 * 
 * SUPPORTED DATASETS:
 * ==================
 * 1. Binary Classification (Primary):
 *    - Target: 2 classes → [0, 1] or [Yes, No] or [True, False]
 *    - Examples: Disease (Yes/No), Loan Approval, Email Spam, Customer Churn
 *    - Datasets: Titanic, Heart Disease, Breast Cancer, Bank Marketing
 * 
 * 2. Multi-class Classification (Extended):
 *    - Target: 3+ classes → [0, 1, 2, ...] or [A, B, C, ...]
 *    - Examples: Iris Species, Product Categories, Customer Segments
 *    - Datasets: Iris, Wine Quality, Digits (0-9), Multi-class Sentiment
 * 
 * 3. Feature Types Supported:
 *    - Numeric: int, float, any range
 *    - Categorical: string, object (auto-encoded via One-Hot/Label/Target)
 *    - Mixed: Both numeric + categorical columns together
 *    - Handles: Missing values, outliers, duplicates, imbalanced classes
 * 
 * KEY FEATURES:
 * =============
 * - Model Config: Solver (5 types), Penalty (L1/L2/ElasticNet), C value, Max iterations, Random seed
 * - Class Imbalance: SMOTE, Over/Under-sampling, Class weights (Balanced/Custom)
 * - Advanced: Probability threshold adjustment, Stratified splits, Multi-class strategy (OvR/Multinomial)
 * - Graphs: 15+ including Confusion Matrix, ROC, PR Curve, Calibration Curve, Decision Boundary
 * - Metrics: Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, Log Loss
 * - CV: Stratified K-Fold with per-fold metrics
 * - Prediction: Probability outputs, Custom threshold, Batch prediction, CSV export
 */

import React, { useState } from "react";
import { FaPlay, FaSpinner } from "react-icons/fa";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

interface LogisticRegressionProps {
    projectName: string;
    projectAlgo: string;
    projectTime: string;
    projectId: string;
}

const LogisticRegressionComponent: React.FC<LogisticRegressionProps> = ({ projectName, projectAlgo, projectTime, projectId }) => {
    const router = useRouter();
    const [isRunning, setIsRunning] = useState<boolean>(false);

    const handleRunScript = () => {
        console.log("Run script for Logistic Regression");
    };

    return (
        <div>
            <div className="text-xl">
                {/* Tabs Wraps Everything Now */}
                <Tabs defaultValue="home">
                    {/* Project Title & Tabs in One Row */}
                    <div className="flex items-center justify-between px-4 mt-2">
                        <div className="font-bold flex items-center gap-3">
                            <Button
                                onClick={() => router.push('/')}
                                className="rounded-xl border-2 border-[rgb(61,68,77)] bg-white dark:bg-[#0E0E0E] hover:bg-gray-100 dark:hover:bg-[#1a1a1a] text-black dark:text-white shadow-md"
                                title="Back to Home"
                            >
                                ← Back
                            </Button>
                            <h1 className="italic text-2xl">
                                {projectName} - {projectAlgo}{" "}
                                <span className="text-sm lowercase">{projectTime}</span>
                            </h1>
                        </div>

                        {/* Tabs Navigation */}
                        <TabsList className="flex w-[50%] text-black dark:text-white bg-[#e6e6e6] dark:bg-[#0F0F0F]">
                            <TabsTrigger
                                className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="home"
                            >
                                Home
                            </TabsTrigger>
                            <TabsTrigger
                                className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="graphs"
                            >
                                Graphs
                            </TabsTrigger>
                            <TabsTrigger
                                className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="result"
                            >
                                Results
                            </TabsTrigger>
                            <TabsTrigger
                                className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="terminal"
                            >
                                Terminal
                            </TabsTrigger>
                            <TabsTrigger
                                className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628] disabled:opacity-50 disabled:cursor-not-allowed"
                                value="predict"
                                disabled={true}
                            >
                                🔒 Predict
                            </TabsTrigger>
                        </TabsList>

                        <div className="flex gap-2">
                            <Button className="rounded-xl" onClick={handleRunScript} disabled={isRunning}>
                                {isRunning ? <FaSpinner className="animate-spin" /> : <FaPlay />}
                            </Button>
                            
                            <Button 
                                className="rounded-xl border-2 border-red-500 dark:border-red-600 bg-white dark:bg-[#0E0E0E] hover:bg-red-50 dark:hover:bg-red-950 text-black dark:text-white shadow-md" 
                                onClick={() => console.log("Reset")}
                            >
                                🔄 Reset
                            </Button>
                        </div>
                    </div>

                    {/* Tabs Content */}
                    <div className="mt-2">
                        <TabsContent value="home">
                            <div className="border border-[rgb(61,68,77)] flex flex-col gap-3 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4">
                                
                                {/* Dataset Compatibility Info */}
                                <div className="dark:bg-[#1a1d1f] bg-[#f5f5f5] rounded-xl p-4 border-2 border-blue-500 dark:border-blue-600">
                                    <h3 className="text-lg font-bold mb-2 text-center">📋 Supported Dataset Types</h3>
                                    <div className="grid grid-cols-3 gap-4 text-sm">
                                        <div>
                                            <p className="font-semibold text-blue-600 dark:text-blue-400">✅ Binary Classification:</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Target: 2 classes (0/1, Yes/No, True/False)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Examples: Disease prediction, Loan approval, Churn prediction</p>
                                        </div>
                                        <div>
                                            <p className="font-semibold text-green-600 dark:text-green-400">✅ Multi-class Classification:</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Target: 3+ classes (0/1/2..., A/B/C...)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Examples: Iris species, Customer segments, Product categories</p>
                                        </div>
                                        <div>
                                            <p className="font-semibold text-purple-600 dark:text-purple-400">✅ Feature Support:</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Numeric: int, float (any range)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Categorical: string, object (auto-encoded)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Mixed: Both numeric + categorical columns</p>
                                        </div>
                                    </div>
                                </div>

                                {/* First Row - Data Loading */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">📁 Dataset Upload</h3>
                                            <p className="text-sm text-gray-500">Dataset Directory Path</p>
                                            <p className="text-sm text-gray-500">Train Data & Test Data Upload</p>
                                            <p className="text-sm text-gray-500">Test Split Ratio Option</p>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">📊 Select Features</h3>
                                            <p className="text-sm text-gray-500">Select Train Columns</p>
                                            <p className="text-sm text-gray-500">Multi-select with checkboxes</p>
                                            <p className="text-sm text-gray-500">Select All option</p>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">🎯 Target Column</h3>
                                            <p className="text-sm text-gray-500">Select Output Column</p>
                                            <p className="text-sm text-gray-500">Binary Classification (0/1)</p>
                                            <p className="text-sm text-gray-500">Single selection</p>
                                        </div>
                                    </div>
                                </div>

                                {/* Second Row - Exploration & Graphs */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">🔍 Data Exploration</h3>
                                            <p className="text-sm text-gray-500">First/Last 5 Rows</p>
                                            <p className="text-sm text-gray-500">Dataset Shape, Data Types</p>
                                            <p className="text-sm text-gray-500">Missing Values, Duplicates</p>
                                            <p className="text-sm text-gray-500">Class Distribution</p>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">📈 Select Graphs</h3>
                                            <p className="text-sm text-gray-500">Confusion Matrix ⭐</p>
                                            <p className="text-sm text-gray-500">ROC Curve ⭐</p>
                                            <p className="text-sm text-gray-500">Precision-Recall Curve</p>
                                            <p className="text-sm text-gray-500">Feature Importance</p>
                                            <p className="text-sm text-gray-500">Learning Curves</p>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">🧹 Preprocessing</h3>
                                            <p className="text-sm text-gray-500">Handle Missing Values</p>
                                            <p className="text-sm text-gray-500">Remove Duplicates</p>
                                            <p className="text-sm text-gray-500">Outlier Detection</p>
                                        </div>
                                    </div>
                                </div>

                                {/* Third Row - Model Config */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">⚙️ Model Configuration</h3>
                                            <p className="text-sm text-gray-500">Encoding Method (One-Hot, Label, Target)</p>
                                            <p className="text-sm text-gray-500">Solver (lbfgs, liblinear, saga, sag, newton-cg)</p>
                                            <p className="text-sm text-gray-500">Penalty (None, L1, L2, ElasticNet)</p>
                                            <p className="text-sm text-gray-500">C Value (0.001-100) + l1_ratio for ElasticNet</p>
                                            <p className="text-sm text-gray-500">Max Iterations + Random Seed</p>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">⚖️ Class Imbalance</h3>
                                            <p className="text-sm text-gray-500">Enable/Disable Toggle</p>
                                            <p className="text-sm text-gray-500">SMOTE (Synthetic Over-sampling)</p>
                                            <p className="text-sm text-gray-500">Random Over-sampling</p>
                                            <p className="text-sm text-gray-500">Random Under-sampling</p>
                                            <p className="text-sm text-gray-500">Class Weights (None, Balanced, Custom)</p>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">📏 Feature Scaling</h3>
                                            <p className="text-sm text-gray-500">None (Keep Original)</p>
                                            <p className="text-sm text-gray-500">Min-Max Scaling</p>
                                            <p className="text-sm text-gray-500">Standard Scaling (Z-score)</p>
                                            <p className="text-sm text-gray-500">Robust Scaling</p>
                                        </div>
                                    </div>
                                </div>

                                {/* Fourth Row - CV & Advanced */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/2 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">🔄 Cross-Validation</h3>
                                            <p className="text-sm text-gray-500">Enable/Disable CV</p>
                                            <p className="text-sm text-gray-500">Number of Folds (2-20)</p>
                                            <p className="text-sm text-gray-500">Stratified K-Fold (maintains class ratio)</p>
                                            <p className="text-sm text-gray-500">Per-fold Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)</p>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/2 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">🎯 Advanced Options</h3>
                                            <p className="text-sm text-gray-500">Probability Threshold (0.1-0.9, default 0.5)</p>
                                            <p className="text-sm text-gray-500">Stratified Train/Test Split</p>
                                            <p className="text-sm text-gray-500">Multi-class Strategy (OvR/Multinomial)</p>
                                            <p className="text-sm text-gray-500">Target Type Detection (Binary/Multi-class)</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="graphs">
                            <div className="border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4 min-h-[700px]">
                                <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                                    <h2 className="text-4xl font-bold">📊 Graphs Section</h2>
                                    <div className="grid grid-cols-2 gap-4 mt-6">
                                        <div className="space-y-2">
                                            <p className="text-lg font-semibold text-blue-600">⭐ Core Classification Graphs:</p>
                                            <p className="text-sm text-gray-500">✅ Confusion Matrix Heatmap (Primary)</p>
                                            <p className="text-sm text-gray-500">✅ ROC Curve with AUC Score</p>
                                            <p className="text-sm text-gray-500">✅ Precision-Recall Curve</p>
                                            <p className="text-sm text-gray-500">✅ Classification Report Bar Chart</p>
                                            <p className="text-sm text-gray-500">✅ Feature Importance (Coefficient Magnitudes)</p>
                                            <p className="text-sm text-gray-500">✅ Decision Boundary Plot (for 2D features)</p>
                                        </div>
                                        <div className="space-y-2">
                                            <p className="text-lg font-semibold text-green-600">📈 Performance & Analysis:</p>
                                            <p className="text-sm text-gray-500">✅ Learning Curves (Overall + Per Fold)</p>
                                            <p className="text-sm text-gray-500">✅ Prediction Probability Distribution</p>
                                            <p className="text-sm text-gray-500">✅ Class Distribution (Before/After Balancing)</p>
                                            <p className="text-sm text-gray-500">✅ Calibration Curve (Probability Calibration)</p>
                                            <p className="text-sm text-gray-500">✅ Threshold Tuning Curve (Precision-Recall vs Threshold)</p>
                                            <p className="text-sm text-gray-500">✅ Correlation Heatmap</p>
                                        </div>
                                        <div className="space-y-2 col-span-2">
                                            <p className="text-lg font-semibold text-purple-600">📊 Optional Exploratory:</p>
                                            <p className="text-sm text-gray-500">• Box Plots, Histogram Distribution</p>
                                            <p className="text-sm text-gray-500">• Lift Chart, Cumulative Gains Chart</p>
                                            <p className="text-sm text-gray-500">• Multi-class: Per-Class ROC Curves</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="result">
                            <div className="border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4 min-h-[700px]">
                                <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                                    <h2 className="text-4xl font-bold">📋 Results Section</h2>
                                    <div className="grid grid-cols-2 gap-6 mt-6">
                                        <div className="space-y-2">
                                            <p className="text-xl font-semibold text-blue-600">📊 Classification Metrics:</p>
                                            <p className="text-sm text-gray-500">✅ Confusion Matrix (TP, TN, FP, FN)</p>
                                            <p className="text-sm text-gray-500">✅ Accuracy Score (overall correctness)</p>
                                            <p className="text-sm text-gray-500">✅ Precision (positive predictive value)</p>
                                            <p className="text-sm text-gray-500">✅ Recall/Sensitivity (true positive rate)</p>
                                            <p className="text-sm text-gray-500">✅ F1-Score (harmonic mean)</p>
                                            <p className="text-sm text-gray-500">✅ Specificity (true negative rate)</p>
                                            <p className="text-sm text-gray-500">✅ ROC-AUC Score</p>
                                            <p className="text-sm text-gray-500">✅ Log Loss (cross-entropy)</p>
                                        </div>
                                        <div className="space-y-2">
                                            <p className="text-xl font-semibold text-purple-600">🔍 Model Details:</p>
                                            <p className="text-sm text-gray-500">✅ Classification Report Table</p>
                                            <p className="text-sm text-gray-500">✅ Per-Class Metrics (Precision, Recall, F1)</p>
                                            <p className="text-sm text-gray-500">✅ Support (samples per class)</p>
                                            <p className="text-sm text-gray-500">✅ Model Coefficients & Intercept</p>
                                            <p className="text-sm text-gray-500">✅ Feature Ranking by Importance</p>
                                            <p className="text-sm text-gray-500">✅ Training & Prediction Time</p>
                                            <p className="text-sm text-gray-500">✅ Convergence Status</p>
                                        </div>
                                        <div className="space-y-2 col-span-2">
                                            <p className="text-xl font-semibold text-green-600">🔄 Cross-Validation Results:</p>
                                            <p className="text-sm text-gray-500">✅ Per-Fold Metrics Table (Acc, Prec, Rec, F1, AUC)</p>
                                            <p className="text-sm text-gray-500">✅ Mean Scores with Std Dev (Model Stability Indicator)</p>
                                            <p className="text-sm text-gray-500">✅ Best Fold Performance Highlight</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="terminal">
                            <div className="ml-4 mr-4">
                                <div className="border border-[rgb(61,68,77)] h-[640px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl text-sm p-4 overflow-y-auto">
                                    <div className="flex flex-col items-center justify-center h-full text-center">
                                        <h2 className="text-2xl font-bold mb-4">🖥️ Terminal Output</h2>
                                        <p className="text-gray-500">Training logs, data exploration results, and script output will appear here</p>
                                        <p className="text-gray-500 mt-2">Download logs button will be available after training</p>
                                    </div>
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="predict">
                            <div className="border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-8 min-h-[700px]">
                                <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                                    <h2 className="text-4xl font-bold">🔮 Prediction Interface</h2>
                                    <div className="grid grid-cols-2 gap-6 mt-6">
                                        <div className="space-y-2">
                                            <p className="text-xl font-semibold text-blue-600">📝 Input Form:</p>
                                            <p className="text-sm text-gray-500">✅ Dynamic form based on trained features</p>
                                            <p className="text-sm text-gray-500">✅ Categorical Features → Dropdown Select</p>
                                            <p className="text-sm text-gray-500">✅ Numeric Features → Number Input Fields</p>
                                            <p className="text-sm text-gray-500">✅ Input Validation (range, type checking)</p>
                                            <p className="text-sm text-gray-500">✅ Clear/Reset Button</p>
                                            <p className="text-sm text-gray-500">✅ Tooltips showing feature info</p>
                                        </div>
                                        <div className="space-y-2">
                                            <p className="text-xl font-semibold text-green-600">📊 Prediction Output:</p>
                                            <p className="text-sm text-gray-500">✅ Predicted Class (0, 1 or class labels)</p>
                                            <p className="text-sm text-gray-500">✅ Prediction Confidence Bar</p>
                                            <p className="text-sm text-gray-500">✅ Probability Distribution (all classes)</p>
                                            <p className="text-sm text-gray-500">✅ Class 0 Probability: XX.X%</p>
                                            <p className="text-sm text-gray-500">✅ Class 1 Probability: XX.X%</p>
                                            <p className="text-sm text-gray-500">✅ Decision Explanation (threshold-based)</p>
                                        </div>
                                        <div className="space-y-2 col-span-2">
                                            <p className="text-xl font-semibold text-purple-600">⚙️ Model Selection & Options:</p>
                                            <p className="text-sm text-gray-500">✅ Model Dropdown (Base, CV Fold 1-N, Different Penalties)</p>
                                            <p className="text-sm text-gray-500">✅ Custom Threshold Slider (adjust decision boundary)</p>
                                            <p className="text-sm text-gray-500">✅ Batch Prediction (upload CSV for multiple predictions)</p>
                                            <p className="text-sm text-gray-500">✅ Export Predictions (download as CSV)</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </TabsContent>
                    </div>
                </Tabs>
            </div>
        </div>
    );
};

export default LogisticRegressionComponent;
