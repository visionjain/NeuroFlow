"use client";

/**
 * K-NEAREST NEIGHBORS (KNN) - COMPLETE WIREFRAME
 * 
 * SUPPORTED TASKS:
 * ================
 * 1. Classification (Binary & Multi-class)
 *    - Iris Species, Breast Cancer, Wine Quality, Digits
 * 
 * 2. Regression
 *    - House Prices, Stock Prediction, Sales Forecasting
 * 
 * KEY FEATURES:
 * =============
 * - K Value Optimization (Auto-find best K)
 * - 6 Distance Metrics (Euclidean, Manhattan, Minkowski, etc.)
 * - Multiple Algorithms (Ball Tree, KD Tree, Brute Force)
 * - Feature Scaling (MANDATORY - 4 methods)
 * - Cross-Validation Support
 * - Dimensionality Reduction (PCA, t-SNE, UMAP)
 * - 20+ Visualization Graphs
 * - Decision Boundary Plots (2D/3D)
 */

import React, { useState, useEffect, useRef } from "react";
import { FaPlay, FaSpinner } from "react-icons/fa";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

// Info Tooltip Component
const InfoTooltip = ({ title, description }: { title: string; description: string }) => {
    return (
        <Dialog>
            <DialogTrigger asChild>
                <button 
                    className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-white dark:bg-gray-200 border-2 border-gray-300 dark:border-gray-400 hover:border-blue-500 dark:hover:border-blue-400 transition-colors cursor-pointer ml-1"
                    onClick={(e) => e.stopPropagation()}
                >
                    <span className="text-black font-bold text-xs">!</span>
                </button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
                <DialogHeader>
                    <DialogTitle className="text-lg font-bold">{title}</DialogTitle>
                </DialogHeader>
                <div className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                    {description}
                </div>
            </DialogContent>
        </Dialog>
    );
};

interface KNNProps {
    projectName: string;
    projectAlgo: string;
    projectTime: string;
    projectId: string;
}

const KNNComponent: React.FC<KNNProps> = ({ projectName, projectAlgo, projectTime, projectId }) => {
    const router = useRouter();
    const [isRunning, setIsRunning] = useState<boolean>(false);
    
    // File and path states
    const [trainFile, setTrainFile] = useState<string | null>(null);
    const [testFile, setTestFile] = useState<string | null>(null);
    const [datasetPath, setDatasetPath] = useState<string>("");
    const [showTestUpload, setShowTestUpload] = useState(true);
    const trainInputRef = useRef<HTMLInputElement>(null);
    const testInputRef = useRef<HTMLInputElement>(null);
    const terminalRef = useRef<HTMLDivElement>(null);
    
    // Data states
    const [logs, setLogs] = useState<string>("");
    const [testSplitRatio, setTestSplitRatio] = useState<string>("0.2");
    const [trainColumns, setTrainColumns] = useState<string[]>([]);
    const [selectedTrainColumns, setSelectedTrainColumns] = useState<string[]>([]);
    const [selectedOutputColumn, setSelectedOutputColumn] = useState<string | null>(null);
    const [results, setResults] = useState<string>("");
    const [taskType, setTaskType] = useState<string>("classification");
    
    // Preprocessing states
    const [selectedHandlingMissingValue, setSelectedHandlingMissingValue] = useState<string>("Drop Rows with Missing Values");
    const [removeDuplicates, setRemoveDuplicates] = useState(true);
    const [encodingMethod, setEncodingMethod] = useState("one-hot");
    const [selectedFeatureScaling, setSelectedFeatureScaling] = useState<string>("Standard Scaling (Z-score Normalization)");
    
    // Outlier detection states
    const [enableOutlierDetection, setEnableOutlierDetection] = useState(false);
    const [outlierMethod, setOutlierMethod] = useState("");
    const [zScoreThreshold, setZScoreThreshold] = useState(3.0);
    const [iqrLower, setIqrLower] = useState(1.5);
    const [iqrUpper, setIqrUpper] = useState(1.5);
    
    // KNN Configuration states
    const [kValue, setKValue] = useState("5");
    const [enableAutoK, setEnableAutoK] = useState(false);
    const [kRangeStart, setKRangeStart] = useState("1");
    const [kRangeEnd, setKRangeEnd] = useState("20");
    const [distanceMetric, setDistanceMetric] = useState("euclidean");
    const [weights, setWeights] = useState("uniform");
    const [algorithm, setAlgorithm] = useState("auto");
    const [leafSize, setLeafSize] = useState("30");
    const [pValue, setPValue] = useState("2");
    
    // Advanced options
    const [enableCV, setEnableCV] = useState(false);
    const [cvFolds, setCvFolds] = useState("5");
    const [useStratifiedSplit, setUseStratifiedSplit] = useState(true);
    const [randomSeed, setRandomSeed] = useState("42");
    
    // Dimensionality Reduction
    const [enableDimReduction, setEnableDimReduction] = useState(false);
    const [dimReductionMethod, setDimReductionMethod] = useState("pca");
    const [nComponents, setNComponents] = useState("2");
    
    // Class Imbalance
    const [enableImbalance, setEnableImbalance] = useState(false);
    const [imbalanceMethod, setImbalanceMethod] = useState("none");
    
    // Graph and exploration states
    const [selectedGraphs, setSelectedGraphs] = useState<string[]>([]);
    const [selectedExplorations, setSelectedExplorations] = useState<string[]>([]);
    const [generatedGraphs, setGeneratedGraphs] = useState<string[]>([]);
    const [zoomedGraph, setZoomedGraph] = useState<string | null>(null);
    
    // Model states
    const [modelTrained, setModelTrained] = useState<boolean>(false);
    
    // Prediction states
    const [predictionInputs, setPredictionInputs] = useState<{ [key: string]: string }>({});
    const [predictionResult, setPredictionResult] = useState<string | null>(null);
    const [isPredicting, setIsPredicting] = useState<boolean>(false);
    const [categoricalInfo, setCategoricalInfo] = useState<{
        categorical_cols: string[];
        numeric_cols: string[];
        categorical_values: { [key: string]: string[] };
    } | null>(null);
    const [availableModels, setAvailableModels] = useState<any[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>("knn_model.pkl");
    const [missingFilesWarning, setMissingFilesWarning] = useState<string[]>([]);
    
    // Available options
    const availableHandlingMissingValues = [
        "Mean Imputation",
        "Median Imputation",
        "Mode Imputation",
        "Forward/Backward Fill",
        "Drop Rows with Missing Values",
    ];
    
    const availableFeatureScaling = [
        "Standard Scaling (Z-score Normalization)",
        "Min-Max Scaling",
        "Robust Scaling",
        "Max Abs Scaling"
    ];
    
    const availableExplorations = [
        "First 5 Rows",
        "Last 5 Rows",
        "Dataset Shape",
        "Data Types",
        "Summary Statistics",
        "Missing Values",
        "Unique Values Per Column",
        "Duplicate Rows",
        "Min & Max Values",
        "Correlation Matrix",
        "Skewness",
        "Target Column Distribution",
        "Class Distribution",
        "Feature Distribution"
    ];

    // Dynamically generate graph options
    const getAvailableGraphs = () => {
        const commonGraphs = [
            "Correlation Heatmap",
            "Box Plot",
            "Histogram Distribution",
            "Pair Plot (2D)",
        ];
        
        const knnSpecificGraphs = [
            "K Value Optimization Curve",
            "Decision Boundary (2D)",
            "Distance Distribution",
            "K-Distance Graph",
            "Neighbor Influence Heatmap",
        ];
        
        const classificationGraphs = taskType === "classification" ? [
            "Confusion Matrix",
            "ROC Curve",
            "Precision-Recall Curve",
            "Classification Report",
            "Class Probability Distribution",
            "Calibration Curve",
        ] : [];
        
        const regressionGraphs = taskType === "regression" ? [
            "Actual vs Predicted Scatter",
            "Residual Plot",
            "Error Distribution",
            "Q-Q Plot",
        ] : [];
        
        const featureAnalysisGraphs = [
            "Feature Importance (Permutation)",
            "PCA Scatter Plot",
            "t-SNE Visualization",
        ];
        
        const performanceGraphs = [
            "Learning Curve",
            "Validation Curve",
            "Training Time vs K",
        ];
        
        if (enableCV) {
            performanceGraphs.push("Cross-Validation Scores");
        }
        
        return [...knnSpecificGraphs, ...classificationGraphs, ...regressionGraphs, ...featureAnalysisGraphs, ...performanceGraphs, ...commonGraphs];
    };
    
    const availableGraphs = getAvailableGraphs();

    // Placeholder functions (to be implemented)
    const handleRunScript = () => {
        toast.info("KNN training will be implemented in next phase", {
            style: { background: 'blue', color: 'white' }
        });
    };

    return (
        <div className="min-h-screen dark:bg-[#0E0E0E] bg-[#E6E6E6] text-black dark:text-white flex justify-center items-center p-4">
            <div className="w-full h-full max-w-[95%] flex flex-col">
                <Tabs defaultValue="home" className="w-full h-full flex flex-col">
                    <div className="flex justify-between items-center mb-2 px-4">
                        {/* Project Header */}
                        <div className="flex flex-col">
                            <h1 className="text-2xl font-bold flex items-center gap-2">
                                <span className="text-3xl">🎯</span>
                                {projectName}
                                <span className="text-sm font-normal text-gray-500 dark:text-gray-400">
                                    ({projectAlgo})
                                </span>
                            </h1>
                            <h1 className="text-sm text-gray-600 dark:text-gray-400 ml-10">
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
                                disabled={!modelTrained}
                            >
                                {modelTrained ? "🔮 Predict" : "🔒 Predict"}
                            </TabsTrigger>
                        </TabsList>

                        <div className="flex gap-2">
                            <Button className="rounded-xl" onClick={handleRunScript} disabled={isRunning}>
                                {isRunning ? <FaSpinner className="animate-spin" /> : <FaPlay />}
                            </Button>
                            
                            {modelTrained && (
                                <Button 
                                    className="rounded-xl border-2 border-red-500 dark:border-red-600 bg-white dark:bg-[#0E0E0E] hover:bg-red-50 dark:hover:bg-red-950 text-black dark:text-white shadow-md" 
                                    onClick={() => toast.info("Reset functionality coming soon")}
                                >
                                    🔄 Reset
                                </Button>
                            )}
                        </div>
                    </div>

                    {/* Missing Files Warning Banner */}
                    {missingFilesWarning.length > 0 && (
                        <div className="mx-4 mt-2">
                            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 rounded-lg">
                                <div className="flex items-start">
                                    <div className="flex-shrink-0">
                                        <svg className="h-5 w-5 text-yellow-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                        </svg>
                                    </div>
                                    <div className="ml-3 flex-1">
                                        <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                                            ⚠️ Some files are missing or unavailable
                                        </h3>
                                        <div className="mt-2 text-sm text-yellow-700 dark:text-yellow-300">
                                            <p className="mb-1">The following files could not be found:</p>
                                            <ul className="list-disc list-inside space-y-1">
                                                {missingFilesWarning.map((file, idx) => (
                                                    <li key={idx} className="font-mono text-xs">{file}</li>
                                                ))}
                                            </ul>
                                            <p className="mt-2 text-xs italic">
                                                Note: These files may have been moved or deleted. Functionality may be limited.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Tabs Content */}
                    <div className="mt-2">
                        <TabsContent value="home">
                            <div className="border border-[rgb(61,68,77)] flex flex-col gap-3 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4">
                                
                                {/* KNN Info Banner */}
                                <div className="dark:bg-[#1a1d1f] bg-[#f5f5f5] rounded-xl p-4 border-2 border-green-500 dark:border-green-600">
                                    <h3 className="text-lg font-bold mb-2 text-center">🎯 K-Nearest Neighbors (KNN)</h3>
                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                        <div>
                                            <p className="font-semibold mb-1">📊 Supported Tasks:</p>
                                            <ul className="list-disc list-inside space-y-1 ml-2">
                                                <li>Classification (Binary & Multi-class)</li>
                                                <li>Regression (Numeric prediction)</li>
                                            </ul>
                                        </div>
                                        <div>
                                            <p className="font-semibold mb-1">✨ Key Features:</p>
                                            <ul className="list-disc list-inside space-y-1 ml-2">
                                                <li>No training phase (Lazy learner)</li>
                                                <li>6 Distance metrics available</li>
                                                <li>Optimal K finder included</li>
                                                <li>Decision boundary visualization</li>
                                            </ul>
                                        </div>
                                    </div>
                                    <div className="mt-3 p-2 bg-yellow-100 dark:bg-yellow-900/30 rounded border border-yellow-500">
                                        <p className="text-xs">
                                            ⚠️ <strong>Important:</strong> Feature scaling is <strong>MANDATORY</strong> for KNN (distance-based algorithm)
                                        </p>
                                    </div>
                                </div>

                                {/* WIREFRAME CONTENT WILL BE IMPLEMENTED IN PHASES */}
                                <div className="text-center p-8 border-2 border-dashed border-blue-500 rounded-xl">
                                    <h2 className="text-2xl font-bold mb-4">🚧 KNN Wireframe - Under Construction</h2>
                                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                                        Full implementation coming soon with all features:
                                    </p>
                                    <div className="grid grid-cols-3 gap-4 text-sm">
                                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                                            <p className="font-bold">✅ Phase 1</p>
                                            <p className="text-xs">Dataset Upload & Config</p>
                                        </div>
                                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                                            <p className="font-bold">🔄 Phase 2</p>
                                            <p className="text-xs">KNN Algorithm Setup</p>
                                        </div>
                                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                                            <p className="font-bold">📊 Phase 3</p>
                                            <p className="text-xs">Graphs & Predictions</p>
                                        </div>
                                    </div>
                                </div>

                            </div>
                        </TabsContent>

                        {/* Terminal Tab Content */}
                        <TabsContent value="terminal">
                            <div className="ml-4 mr-4">
                                <div
                                    ref={terminalRef}
                                    className="border border-[rgb(61,68,77)] h-[640px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl text-sm p-4 overflow-y-auto"
                                >
                                    <pre className="whitespace-pre-wrap">{logs || "Terminal Output will be shown here."}</pre>
                                </div>
                            </div>
                        </TabsContent>

                        {/* Results Tab Content */}
                        <TabsContent value="result">
                            <div className="ml-4 mr-4 min-h-[640px] border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl p-6">
                                <h2 className="text-2xl font-bold mb-4">📊 Model Results</h2>
                                <p className="text-gray-500 dark:text-gray-400">Results will appear here after training</p>
                            </div>
                        </TabsContent>

                        {/* Graphs Tab Content */}
                        <TabsContent value="graphs">
                            <div className="ml-4 mr-4 min-h-[640px] border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl p-6">
                                <h2 className="text-2xl font-bold mb-4">📈 Visualizations</h2>
                                <p className="text-gray-500 dark:text-gray-400">Graphs will appear here after training</p>
                            </div>
                        </TabsContent>

                        {/* Predict Tab Content */}
                        <TabsContent value="predict">
                            <div className="ml-4 mr-4 min-h-[640px] border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl p-6">
                                <h2 className="text-2xl font-bold mb-4">🔮 Make Predictions</h2>
                                <p className="text-gray-500 dark:text-gray-400">Prediction interface will appear here</p>
                            </div>
                        </TabsContent>
                    </div>
                </Tabs>
            </div>
        </div>
    );
};

export default KNNComponent;
