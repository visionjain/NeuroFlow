"use client";

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
                                            <p className="text-sm text-gray-500">Encoding Method</p>
                                            <p className="text-sm text-gray-500">Solver (lbfgs, liblinear, saga)</p>
                                            <p className="text-sm text-gray-500">Penalty (None, L1, L2, ElasticNet)</p>
                                            <p className="text-sm text-gray-500">C Value (Regularization)</p>
                                            <p className="text-sm text-gray-500">Max Iterations</p>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">⚖️ Class Imbalance</h3>
                                            <p className="text-sm text-gray-500">Enable/Disable</p>
                                            <p className="text-sm text-gray-500">SMOTE (Over-sampling)</p>
                                            <p className="text-sm text-gray-500">Random Over/Under-sampling</p>
                                            <p className="text-sm text-gray-500">Class Weights (Balanced)</p>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">📏 Feature Scaling</h3>
                                            <p className="text-sm text-gray-500">Min-Max Scaling</p>
                                            <p className="text-sm text-gray-500">Standard Scaling (Z-score)</p>
                                            <p className="text-sm text-gray-500">Robust Scaling</p>
                                        </div>
                                    </div>
                                </div>

                                {/* Fourth Row - CV */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-full bg-white p-4 flex items-center justify-center">
                                        <div className="text-center">
                                            <h3 className="text-lg font-semibold mb-2">🔄 Cross-Validation</h3>
                                            <p className="text-sm text-gray-500">Enable/Disable CV</p>
                                            <p className="text-sm text-gray-500">Number of Folds (2-20)</p>
                                            <p className="text-sm text-gray-500">Per-fold Metrics (Accuracy, Precision, Recall, F1)</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="graphs">
                            <div className="border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4 min-h-[700px]">
                                <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                                    <h2 className="text-4xl font-bold">📊 Graphs Section</h2>
                                    <div className="space-y-2 text-lg text-gray-500">
                                        <p>✅ Confusion Matrix Heatmap (Primary)</p>
                                        <p>✅ ROC Curve with AUC Score</p>
                                        <p>✅ Precision-Recall Curve</p>
                                        <p>✅ Classification Report Bar Chart</p>
                                        <p>✅ Feature Importance (Coefficient Magnitudes)</p>
                                        <p>✅ Learning Curves (Overall + Per Fold)</p>
                                        <p>✅ Prediction Probability Distribution</p>
                                        <p>✅ Class Distribution (Before/After Balancing)</p>
                                        <p>• Correlation Heatmap</p>
                                        <p>• Box Plots</p>
                                        <p>• Histogram Distribution</p>
                                    </div>
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="result">
                            <div className="border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4 min-h-[700px]">
                                <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                                    <h2 className="text-4xl font-bold">📋 Results Section</h2>
                                    <div className="space-y-2 text-lg text-gray-500">
                                        <p className="text-xl font-semibold text-blue-600">Classification Metrics:</p>
                                        <p>✅ Confusion Matrix (TP, TN, FP, FN)</p>
                                        <p>✅ Accuracy Score</p>
                                        <p>✅ Precision, Recall, F1-Score</p>
                                        <p>✅ ROC-AUC Score</p>
                                        <p>✅ Classification Report (Per-Class Metrics)</p>
                                        <p>✅ Model Coefficients & Intercept</p>
                                        <p>✅ Training Time</p>
                                        <p className="text-xl font-semibold text-green-600 mt-4">With Cross-Validation:</p>
                                        <p>✅ Per-Fold Metrics Table</p>
                                        <p>✅ Mean Scores (Accuracy, Precision, Recall, F1)</p>
                                        <p>✅ Standard Deviation (Model Stability)</p>
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
                                    <div className="space-y-2 text-lg text-gray-500">
                                        <p>✅ Input Form for Feature Values</p>
                                        <p>✅ Categorical Dropdowns (if applicable)</p>
                                        <p>✅ Numeric Input Fields</p>
                                        <p className="text-xl font-semibold text-purple-600 mt-4">Output Display:</p>
                                        <p>✅ Predicted Class (0 or 1)</p>
                                        <p>✅ Probability of Class 0</p>
                                        <p>✅ Probability of Class 1</p>
                                        <p>✅ Confidence Score</p>
                                        <p>✅ Model Selection Dropdown</p>
                                        <p className="text-sm mt-4">(Base model, CV fold models, different regularizations)</p>
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
