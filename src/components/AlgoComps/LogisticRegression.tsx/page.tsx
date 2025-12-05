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

interface LogisticRegressionProps {
    projectName: string;
    projectAlgo: string;
    projectTime: string;
    projectId: string;
}

const LogisticRegressionComponent: React.FC<LogisticRegressionProps> = ({ projectName, projectAlgo, projectTime, projectId }) => {
    const router = useRouter();
    const [isRunning, setIsRunning] = useState<boolean>(false);
    
    // File and path states
    const [trainFile, setTrainFile] = useState<string | null>(null);
    const [testFile, setTestFile] = useState<string | null>(null);
    const [datasetPath, setDatasetPath] = useState<string>("");
    const [showTestUpload, setShowTestUpload] = useState(true);
    const trainInputRef = useRef<HTMLInputElement>(null);
    const testInputRef = useRef<HTMLInputElement>(null);
    const terminalRef = useRef<HTMLPreElement>(null);
    
    // Data states
    const [logs, setLogs] = useState<string>("");
    const [testSplitRatio, setTestSplitRatio] = useState<string>("0.2");
    const [trainColumns, setTrainColumns] = useState<string[]>([]);
    const [selectedTrainColumns, setSelectedTrainColumns] = useState<string[]>([]);
    const [selectedOutputColumn, setSelectedOutputColumn] = useState<string | null>(null);
    const [results, setResults] = useState<string>("");
    
    // Preprocessing states
    const [selectedHandlingMissingValue, setSelectedHandlingMissingValue] = useState<string>("Drop Rows with Missing Values");
    const [removeDuplicates, setRemoveDuplicates] = useState(true);
    const [encodingMethod, setEncodingMethod] = useState("one-hot");
    const [selectedFeatureScaling, setSelectedFeatureScaling] = useState<string | null>(null);
    
    // Model configuration states
    const [solver, setSolver] = useState("lbfgs");
    const [penalty, setPenalty] = useState("none");
    const [cValue, setCValue] = useState("1.0");
    const [maxIter, setMaxIter] = useState("300");
    const [randomSeed, setRandomSeed] = useState("42");
    const [l1Ratio, setL1Ratio] = useState("0.5");
    
    // Class imbalance states
    const [enableImbalance, setEnableImbalance] = useState(false);
    const [imbalanceMethod, setImbalanceMethod] = useState<string>("none");
    const [classWeight, setClassWeight] = useState<string>("none");
    
    // Advanced options
    const [probabilityThreshold, setProbabilityThreshold] = useState("0.5");
    const [useStratifiedSplit, setUseStratifiedSplit] = useState(true);
    const [multiClassStrategy, setMultiClassStrategy] = useState("auto");
    
    // Cross-validation states
    const [enableCV, setEnableCV] = useState(false);
    const [cvFolds, setCvFolds] = useState("5");
    
    // Graph and exploration states
    const [selectedGraphs, setSelectedGraphs] = useState<string[]>([]);
    const [selectedExplorations, setSelectedExplorations] = useState<string[]>([]);
    const [generatedGraphs, setGeneratedGraphs] = useState<string[]>([]);
    const [selectedEffectFeatures, setSelectedEffectFeatures] = useState<string[]>([]);
    const [zoomedGraph, setZoomedGraph] = useState<string | null>(null);
    
    // Model states
    const [modelTrained, setModelTrained] = useState<boolean>(false);
    
    // Prediction states
    const [predictionInputs, setPredictionInputs] = useState<{ [key: string]: string }>({});
    const [predictionResult, setPredictionResult] = useState<string | null>(null);
    const [predictionIsBinary, setPredictionIsBinary] = useState<boolean | null>(null);
    const [isPredicting, setIsPredicting] = useState<boolean>(false);
    const [categoricalInfo, setCategoricalInfo] = useState<{
        categorical_cols: string[];
        numeric_cols: string[];
        categorical_values: { [key: string]: string[] };
    } | null>(null);
    const [availableModels, setAvailableModels] = useState<any[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>("logistic_model.pkl");
    const [missingFilesWarning, setMissingFilesWarning] = useState<string[]>([]);
    
    // Available options
    const availableHandlingMissingValues = [
        "Fill with Mean",
        "Fill with Median",
        "Fill with Mode",
        "Forward/Backward Fill",
        "Drop Rows with Missing Values",
    ];
    
    const availableFeatureScaling = [
        "Min-Max Scaling",
        "Standard Scaling (Z-score Normalization)",
        "Robust Scaling"
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
        "Class Distribution"
    ];

    // Dynamically generate graph options based on CV settings
    const getAvailableGraphs = () => {
        // Add learning curve options at the top
        const learningCurves = ["Learning Curve - Overall"];
        if (enableCV) {
            learningCurves.push("Learning Curve - All Folds");
        }
        
        const restGraphs = [
            "Confusion Matrix",
            "ROC Curve",
            "Precision-Recall Curve",
            "Feature Importance",
            "Classification Report",
            "Probability Distribution",
            "Calibration Curve",
            "Decision Boundary (2D)",
            "Class Separation (PCA)",
            "Correlation Heatmap",
            "Box Plot",
            "Histogram Distribution",
            "Individual Effect Plot",
            "Mean Effect Plot",
            "Trend Effect Plot",
            "Shap Summary Plot",
        ];
        
        return [...learningCurves, ...restGraphs];
    };
    
    const availableGraphs = getAvailableGraphs();

    // Load project state on mount
    useEffect(() => {
        const loadProjectState = async () => {
            try {
                const response = await fetch(`/api/users/projectstate?projectId=${projectId}`);
                const data = await response.json();
                
                if (data.hasState && !data.isCorrupted) {
                    const state = data.state;
                    
                    // Restore all state
                    setTrainFile(state.trainFile || null);
                    setTestFile(state.testFile || null);
                    setDatasetPath(state.datasetPath || "");
                    setTrainColumns(state.trainColumns || []);
                    setSelectedTrainColumns(state.selectedTrainColumns || []);
                    setSelectedOutputColumn(state.selectedOutputColumn || null);
                    setTestSplitRatio(state.testSplitRatio || "0.2");
                    setSelectedHandlingMissingValue(state.selectedHandlingMissingValue || "Drop Rows with Missing Values");
                    setRemoveDuplicates(state.removeDuplicates ?? true);
                    setEncodingMethod(state.encodingMethod || "one-hot");
                    setSelectedFeatureScaling(state.selectedFeatureScaling || null);
                    setSolver(state.solver || "lbfgs");
                    setPenalty(state.penalty || "none");
                    setCValue(state.cValue || "1.0");
                    setMaxIter(state.maxIter || "300");
                    setRandomSeed(state.randomSeed || "42");
                    setL1Ratio(state.l1Ratio || "0.5");
                    setEnableImbalance(state.enableImbalance ?? false);
                    setImbalanceMethod(state.imbalanceMethod || "none");
                    setClassWeight(state.classWeight || "none");
                    setProbabilityThreshold(state.probabilityThreshold || "0.5");
                    setUseStratifiedSplit(state.useStratifiedSplit ?? true);
                    setMultiClassStrategy(state.multiClassStrategy || "auto");
                    setEnableCV(state.enableCV ?? false);
                    setCvFolds(state.cvFolds || "5");
                    setSelectedGraphs(state.selectedGraphs || []);
                    setSelectedExplorations(state.selectedExplorations || []);
                    setSelectedEffectFeatures(state.selectedEffectFeatures || []);
                    setLogs(state.logs || "");
                    setResults(state.results || "");
                    setGeneratedGraphs(state.generatedGraphs || []);
                    setModelTrained(state.modelTrained ?? false);
                    setAvailableModels(state.availableModels || []);
                    
                    toast.success('Project restored successfully', {
                        style: {
                            background: 'green',
                            color: 'white',
                        },
                    });
                    
                    // Combine critical missing files and warnings
                    const allWarnings: string[] = [];
                    
                    if (data.warnings && data.warnings.length > 0) {
                        console.warn(`⚠️ Dataset files missing:`, data.warnings);
                        allWarnings.push(...data.warnings);
                    }
                    
                    if (data.missingFiles && data.missingFiles.length > 0) {
                        console.warn(`⚠️ Some files are missing:`, data.missingFiles);
                        allWarnings.push(...data.missingFiles);
                    }
                    
                    setMissingFilesWarning(allWarnings);
                } else if (data.isCorrupted) {
                    console.error("❌ Project is corrupted:", data.missingFiles);
                    alert(
                        `❌ Project Data Corrupted\n\n` +
                        `Critical files are missing:\n• ${data.missingFiles.join("\n• ")}\n\n` +
                        `Please run the training again to restore the project.`
                    );
                    // Reset to clean state
                    setModelTrained(false);
                    setGeneratedGraphs([]);
                    setResults("");
                    setLogs("");
                } else {
                    console.log("ℹ️ No saved state found - starting fresh");
                }
            } catch (error) {
                console.error("Error loading project state:", error);
            }
        };

        loadProjectState();
    }, [projectId]);

    // Auto-scroll terminal to bottom
    useEffect(() => {
        if (terminalRef.current) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [logs]);

    // Fetch categorical info when model is trained
    useEffect(() => {
        if (modelTrained && datasetPath && trainFile) {
            const fetchCategoricalInfo = async () => {
                try {
                    const normalizedPath = datasetPath.trim().replace(/[/\\]+$/, "");
                    const isWindows = navigator.platform.startsWith("Win");
                    const separator = isWindows ? "\\\\" : "/";
                    const modelDir = `${normalizedPath}${separator}logistic-${trainFile?.split(".")[0]}`;
                    const modelPath = `${modelDir}${separator}logistic_model.pkl`;

                    const response = await fetch(`/api/users/scripts/predict?model_path=${encodeURIComponent(modelPath)}`);
                    const data = await response.json();

                    if (data.categorical_cols && data.categorical_values) {
                        setCategoricalInfo({
                            categorical_cols: data.categorical_cols,
                            numeric_cols: data.numeric_cols || [],
                            categorical_values: data.categorical_values
                        });
                    }

                    // Also fetch available models
                    if (data.available_models) {
                        setAvailableModels(data.available_models);
                        // Set default to final model
                        setSelectedModel("logistic_model.pkl");
                        
                        // Save state again now that we have available models (silent save)
                        setTimeout(() => {
                            saveProjectState(undefined, true);
                        }, 500);
                    }
                } catch (error) {
                    console.error("Failed to fetch categorical info:", error);
                }
            };

            fetchCategoricalInfo();
        }
    }, [modelTrained, datasetPath, trainFile]);
    
    // Save project state to database
    const saveProjectState = async (overrideState?: any, silent: boolean = false) => {
        try {
            const state = overrideState || {
                trainFile,
                testFile,
                datasetPath,
                trainColumns,
                selectedTrainColumns,
                selectedOutputColumn,
                testSplitRatio,
                selectedHandlingMissingValue,
                removeDuplicates,
                encodingMethod,
                selectedFeatureScaling,
                solver,
                penalty,
                cValue,
                maxIter,
                randomSeed,
                l1Ratio,
                enableImbalance,
                imbalanceMethod,
                classWeight,
                probabilityThreshold,
                useStratifiedSplit,
                multiClassStrategy,
                enableCV,
                cvFolds,
                selectedGraphs,
                selectedExplorations,
                selectedEffectFeatures,
                logs,
                results,
                generatedGraphs,
                modelTrained: true,
                availableModels
            };
            
            const response = await fetch('/api/users/projectstate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ projectId, state })
            });

            const data = await response.json();
            
            if (data.success && !silent) {
                toast.success('Project saved successfully', {
                    style: {
                        background: 'green',
                        color: 'white',
                    },
                });
            } else if (!data.success) {
                console.error("Failed to save project state:", data.error);
            }
        } catch (error) {
            console.error("Error saving project state:", error);
        }
    };
    
    const handleRunScript = () => {
        // Validate inputs
        if (!datasetPath && !trainFile) {
            alert("Please select a dataset path or upload a training file");
            setIsRunning(false);
            return;
        }
        
        if (selectedTrainColumns.length === 0) {
            alert("Please select at least one feature column");
            setIsRunning(false);
            return;
        }
        
        if (!selectedOutputColumn) {
            alert("Please select a target column");
            setIsRunning(false);
            return;
        }
        
        setIsRunning(true);
        setLogs("");
        setResults("");
        setGeneratedGraphs([]);
        setModelTrained(false);

        // Normalize datasetPath to remove trailing slashes
        let normalizedPath = datasetPath.trim();
        if (normalizedPath.endsWith("\\") || normalizedPath.endsWith("/")) {
            normalizedPath = normalizedPath.slice(0, -1);
        }

        // Detect OS to use appropriate separator
        const isWindows = navigator.platform.startsWith("Win");
        const separator = isWindows ? "\\\\" : "/";

        // Construct file paths
        const train_csv_path = `${normalizedPath}${separator}${trainFile}`;
        const test_csv_path = testFile ? `${normalizedPath}${separator}${testFile}` : "None";

        // Prepare API query parameters
        const queryParams = new URLSearchParams({
            train_csv_path,
            test_csv_path,
            train_columns: JSON.stringify(selectedTrainColumns),
            output_column: selectedOutputColumn,
            selected_graphs: JSON.stringify(selectedGraphs),
            selected_missingval_tech: JSON.stringify(selectedHandlingMissingValue),
            remove_Duplicates: JSON.stringify(removeDuplicates),
            encoding_Method: encodingMethod,
            feature_scaling: selectedFeatureScaling ? selectedFeatureScaling : "",
            available_Explorations: JSON.stringify(selectedExplorations),
            effect_features: JSON.stringify(selectedEffectFeatures),
            
            // Logistic Regression specific
            solver: solver,
            penalty: penalty,
            c_value: cValue,
            max_iter: maxIter || "0",
            random_seed: randomSeed,
            ...(penalty === "elasticnet" && { l1_ratio: l1Ratio }),
            
            // Class imbalance
            enable_imbalance: JSON.stringify(enableImbalance),
            ...(enableImbalance && {
                imbalance_method: imbalanceMethod === 'none' ? '' : imbalanceMethod,
                class_weight: classWeight,
            }),
            
            // Advanced options
            probability_threshold: probabilityThreshold,
            use_stratified_split: JSON.stringify(useStratifiedSplit),
            multi_class_strategy: multiClassStrategy,
            
            // Cross-validation
            enable_cv: JSON.stringify(enableCV),
            cv_folds: cvFolds,
        });

        if (!testFile && testSplitRatio) {
            queryParams.append("test_split_ratio", testSplitRatio);
        }

        const apiUrl = `/api/users/scripts/logisticregression?${queryParams.toString()}`;

        // Local variable to accumulate all output lines
        let allLogs = "";
        let trainingSuccessful = false;
        const eventSource = new EventSource(apiUrl);

        eventSource.onmessage = (event) => {
            if (event.data === "END_OF_STREAM") {
                // Check if training completed successfully
                trainingSuccessful = allLogs.includes("FINISHED SUCCESSFULLY") || allLogs.includes("Training completed");
                
                // Extract results section
                let resultsText = "";
                
                const hasResults = allLogs.includes("CLASSIFICATION METRICS") || allLogs.includes("CLASSIFICATION RESULTS");
                
                if (hasResults) {
                    const searchString = allLogs.includes("CLASSIFICATION METRICS") ? "CLASSIFICATION METRICS" : "CLASSIFICATION RESULTS";
                    const startIdx = allLogs.indexOf(searchString);
                    const endIdx = allLogs.indexOf("FINISHED SUCCESSFULLY", startIdx);
                    
                    if (startIdx !== -1) {
                        const extractedSection = endIdx !== -1 
                            ? allLogs.substring(startIdx, endIdx).trim()
                            : allLogs.substring(startIdx).trim();
                        
                        const lines = allLogs.split("\n");
                        const summaryLineIdx = lines.findIndex(line => line.includes(searchString));
                        if (summaryLineIdx >= 0) {
                            const startLineIdx = Math.max(0, summaryLineIdx - 1);
                            const endLineIdx = endIdx !== -1 
                                ? lines.findIndex(line => line.includes("FINISHED SUCCESSFULLY"))
                                : lines.length;
                            resultsText = lines.slice(startLineIdx, endLineIdx).join("\n");
                        } else {
                            resultsText = extractedSection;
                        }
                    }
                }
                
                // Fallback to line-by-line extraction
                if (!resultsText) {
                    const resultLines = allLogs
                        .split("\n")
                        .filter(
                            (line) =>
                                line.startsWith("Accuracy:") ||
                                line.startsWith("Precision:") ||
                                line.startsWith("Recall:") ||
                                line.startsWith("F1-Score:") ||
                                line.startsWith("ROC-AUC:") ||
                                line.includes("Confusion Matrix") ||
                                line.includes("Classification Report")
                        );
                    resultsText = resultLines.join("\n");
                }
                
                setResults(resultsText);
                
                // Parse generated graphs JSON
                let parsedGraphs: string[] = [];
                const graphsMatch = allLogs.match(/__GENERATED_GRAPHS_JSON__(.+?)__END_GRAPHS__/);
                if (graphsMatch) {
                    try {
                        parsedGraphs = JSON.parse(graphsMatch[1]);
                        setGeneratedGraphs(parsedGraphs);
                    } catch (e) {
                        console.error("Failed to parse graphs JSON:", e);
                    }
                }
                
                eventSource.close();
                setIsRunning(false);
                
                // Enable prediction tab if training was successful
                if (trainingSuccessful) {
                    setModelTrained(true);
                    toast.success("Training completed successfully!", {
                        style: { background: 'green', color: 'white' }
                    });
                    
                    // Save project state to database with current values
                    saveProjectState({
                        trainFile,
                        testFile,
                        datasetPath,
                        trainColumns,
                        selectedTrainColumns,
                        selectedOutputColumn,
                        testSplitRatio,
                        selectedHandlingMissingValue,
                        removeDuplicates,
                        encodingMethod,
                        selectedFeatureScaling,
                        solver,
                        penalty,
                        cValue,
                        maxIter,
                        randomSeed,
                        l1Ratio,
                        enableImbalance,
                        imbalanceMethod,
                        classWeight,
                        probabilityThreshold,
                        useStratifiedSplit,
                        multiClassStrategy,
                        enableCV,
                        cvFolds,
                        selectedGraphs,
                        selectedExplorations,
                        selectedEffectFeatures,
                        logs: allLogs,
                        results: resultsText,
                        generatedGraphs: parsedGraphs,
                        modelTrained: true,
                        availableModels
                    });
                } else {
                    setModelTrained(false);
                    setLogs((prev) => prev + "\n❌ Training failed. Prediction tab remains locked.\n");
                }
            } else {
                // Filter out the JSON marker from terminal display
                if (!event.data.includes("__GENERATED_GRAPHS_JSON__")) {
                    allLogs += event.data + "\n";
                    setLogs((prev) => prev + event.data + "\n");
                } else {
                    // Still accumulate for parsing, just don't display
                    allLogs += event.data + "\n";
                }
            }
        };

        eventSource.onerror = (error) => {
            console.error("EventSource failed:", error);
            eventSource.close();
            setIsRunning(false);
            setModelTrained(false);
            setLogs((prev) => prev + "\n❌ Connection error or training interrupted.\n");
        };
    };
    
    // File selection handler
    const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>, type: string) => {
        const file = event.target.files?.[0];
        if (file) {
            const fileName = file.name;
            if (type === "Train") {
                setTrainFile(fileName);
                // Read columns from train file
                await readColumnsFromFile(file);
            } else {
                setTestFile(fileName);
            }
        }
    };
    
    // Read columns from CSV/Excel file
    const readColumnsFromFile = async (file: File) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target?.result as string;
            const firstLine = text.split('\n')[0];
            // Remove quotes from column names if present (handles both single and double quotes)
            const columns = firstLine.split(',').map(col => 
                col.trim().replace(/^["']|["']$/g, '').replace(/\r/g, '')
            );
            setTrainColumns(columns);
        };
        reader.readAsText(file);
    };
    
    // Toggle test dataset input
    const toggleTestDataset = () => {
        setShowTestUpload(!showTestUpload);
        if (!showTestUpload) {
            setTestFile(null);
        }
    };
    
    // Test split ratio handlers
    const handleTestSplitChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        let value = event.target.value;
        if (/^\d*\.?\d{0,2}$/.test(value) || value === "") {
            setTestSplitRatio(value);
        }
    };
    
    const handleTestSplitBlur = () => {
        let numValue = parseFloat(testSplitRatio);
        if (isNaN(numValue) || numValue < 0.01 || numValue > 0.99) {
            setTestSplitRatio("0.2");
            toast.error("Test split ratio must be between 0.01 and 0.99", {
                style: { background: 'red', color: 'white' }
            });
        }
    };
    
    // Column selection handlers
    const toggleTrainColumn = (col: string) => {
        setSelectedTrainColumns(prev =>
            prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
        );
    };
    
    const toggleSelectAll = () => {
        if (selectedTrainColumns.length === trainColumns.length) {
            setSelectedTrainColumns([]);
        } else {
            setSelectedTrainColumns([...trainColumns]);
        }
    };
    
    const handleOutputColumnSelect = (col: string) => {
        setSelectedOutputColumn(col);
    };
    
    // Exploration toggles
    const toggleExploration = (technique: string) => {
        setSelectedExplorations(prev =>
            prev.includes(technique) ? prev.filter(t => t !== technique) : [...prev, technique]
        );
    };
    
    const toggleSelectAllExplorations = () => {
        if (selectedExplorations.length === availableExplorations.length) {
            setSelectedExplorations([]);
        } else {
            setSelectedExplorations([...availableExplorations]);
        }
    };
    
    // Graph toggles
    const toggleGraph = (graph: string) => {
        setSelectedGraphs(prev =>
            prev.includes(graph) ? prev.filter(g => g !== graph) : [...prev, graph]
        );
    };
    
    const toggleSelectAllGraphs = () => {
        if (selectedGraphs.length === availableGraphs.length) {
            setSelectedGraphs([]);
        } else {
            setSelectedGraphs([...availableGraphs]);
        }
    };

    // Effect features toggle
    const toggleEffectFeature = (col: string) => {
        setSelectedEffectFeatures(prev =>
            prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
        );
    };
    
    const toggleSelectAllEffectFeatures = () => {
        if (selectedEffectFeatures.length === selectedTrainColumns.length) {
            setSelectedEffectFeatures([]);
        } else {
            setSelectedEffectFeatures([...selectedTrainColumns]);
        }
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
                                    onClick={async () => {
                                        if (confirm("Are you sure you want to reset this project? All saved state will be cleared.")) {
                                            try {
                                                const response = await fetch(`/api/users/projectstate?projectId=${projectId}`, {
                                                    method: 'DELETE'
                                                });
                                                const data = await response.json();
                                                if (data.success) {
                                                    // Reset all state to initial values
                                                    setTrainFile(null);
                                                    setTestFile(null);
                                                    setDatasetPath("");
                                                    setTrainColumns([]);
                                                    setSelectedTrainColumns([]);
                                                    setSelectedOutputColumn(null);
                                                    setLogs("");
                                                    setResults("");
                                                    setGeneratedGraphs([]);
                                                    setModelTrained(false);
                                                    setAvailableModels([]);
                                                    setPredictionResult(null);
                                                    alert("✅ Project reset successfully!");
                                                    window.location.reload();
                                                }
                                            } catch (error) {
                                                console.error("Error resetting project:", error);
                                                alert("Failed to reset project");
                                            }
                                        }
                                    }}
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

                                {/* First Row */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4">
                                        {/* Dataset Directory Path Input */}
                                        <div className="mb-4 text-center">
                                            <Label className="text-sm font-semibold">Dataset Directory Path</Label>
                                            <Input
                                                type="text"
                                                placeholder="Ex: D:\datasetpath"
                                                className="mt-1 dark:bg-[#0F0F0F]"
                                                value={datasetPath}
                                                onChange={(e) => setDatasetPath(e.target.value)}
                                            />
                                        </div>

                                        {/* Train Data Selection & Dynamic Test Handling */}
                                        <div className="flex w-full gap-2">
                                            {/* Train Data Upload */}
                                            <div className="flex flex-col w-full items-center">
                                                <Label className="text-sm font-semibold mb-1">Train Data</Label>
                                                <input
                                                    type="file"
                                                    id="trainDataset"
                                                    accept=".csv, .xlsx"
                                                    ref={trainInputRef}
                                                    onChange={(e) => handleFileSelect(e, "Train")}
                                                    hidden
                                                />
                                                <Button
                                                    className="h-12 w-full flex justify-center items-center border-2 border-dashed border-gray-500 rounded-md 
                                                    hover:bg-gray-100 hover:text-black dark:hover:bg-gray-800 dark:hover:border-gray-300 transition"
                                                    onClick={() => trainInputRef.current?.click()}
                                                >
                                                    {trainFile ? (
                                                        <span className="text-sm truncate w-full text-center">{trainFile}</span>
                                                    ) : (
                                                        <span className="text-3xl">+</span>
                                                    )}
                                                </Button>
                                            </div>

                                            {/* Conditional UI: Test File Upload OR Test Set Split Ratio */}
                                            <div className="flex flex-col w-full items-center">
                                                {showTestUpload ? (
                                                    // Test File Selection
                                                    <>
                                                        <Label className="text-sm font-semibold mb-1">Test Data</Label>
                                                        <input
                                                            type="file"
                                                            id="testDataset"
                                                            accept=".csv, .xlsx"
                                                            ref={testInputRef}
                                                            onChange={(e) => handleFileSelect(e, "Test")}
                                                            hidden
                                                        />
                                                        <Button
                                                            className="h-12 w-full flex justify-center items-center border-2 border-dashed border-gray-500 rounded-md 
                                                            hover:bg-gray-100 hover:text-black dark:hover:bg-gray-800 dark:hover:border-gray-300 transition"
                                                            onClick={() => testInputRef.current?.click()}
                                                        >
                                                            {testFile ? (
                                                                <span className="text-sm truncate w-full text-center">{testFile}</span>
                                                            ) : (
                                                                <span className="text-3xl">+</span>
                                                            )}
                                                        </Button>
                                                    </>
                                                ) : (
                                                    // Test Set Split Ratio Input
                                                    <div className="flex flex-col w-full items-center">
                                                        <Label className="text-sm font-semibold mb-1">Test Set Split Ratio</Label>
                                                        <Input
                                                            type="text"
                                                            value={testSplitRatio}
                                                            onChange={handleTestSplitChange}
                                                            onBlur={handleTestSplitBlur}
                                                            placeholder="Ex: 0.2"
                                                            className="w-full h-12 text-center dark:bg-[#0F0F0F] border border-gray-500 rounded-md"
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                        </div>

                                        {/* Toggle Link */}
                                        <p className="underline mt-2 flex justify-center text-sm text-blue-600 cursor-pointer" onClick={toggleTestDataset}>
                                            {showTestUpload ? "Don't have a test dataset?" : "Have a test dataset?"}
                                        </p>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm flex items-center">
                                                Select Train Columns
                                                <InfoTooltip 
                                                    title="Train Columns" 
                                                    description="Choose which columns from your dataset to use as input features for training the model. These are the variables the model will learn from to make predictions." 
                                                />
                                            </div>

                                            {/* Show "Select All" Checkbox only if a file is selected */}
                                            {trainFile && (
                                                <div className="flex items-center">
                                                    <Checkbox
                                                        checked={selectedTrainColumns.length === trainColumns.length}
                                                        onCheckedChange={() => toggleSelectAll()}
                                                    />
                                                    <span className="ml-1 text-xs">Select All</span>
                                                </div>
                                            )}
                                        </div>

                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <div className="grid grid-cols-2 gap-1">
                                                    {trainColumns.map((col, index) => (
                                                        <div key={index} className="flex items-center text-xs">
                                                            <Checkbox
                                                                checked={selectedTrainColumns.includes(col)}
                                                                onCheckedChange={() => toggleTrainColumn(col)}
                                                            />
                                                            <span className="ml-1">{col}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            ) : (
                                                <div className="text-center">Please select a train file to enable column selection.</div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-1 mt-1 flex items-center">
                                            Select Output Column
                                            <InfoTooltip 
                                                title="Output Column" 
                                                description="Choose the target column (dependent variable) that you want to predict. This should be a categorical column with classes like 0/1, Yes/No, or multiple categories." 
                                            />
                                        </div>

                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <div className="grid grid-cols-2 gap-1">
                                                    {trainColumns.map((col, index) => (
                                                        <div key={index} className="flex items-center text-xs">
                                                            <Checkbox
                                                                checked={selectedOutputColumn === col}
                                                                onCheckedChange={() => handleOutputColumnSelect(col)}
                                                            />
                                                            <span className="ml-1">{col}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            ) : (
                                                <div className="text-center">Please select a train file to select output column.</div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Second Row */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm flex items-center">
                                                Select Data Exploration Techniques
                                                <InfoTooltip 
                                                    title="Data Exploration" 
                                                    description="Explore your data before training. These techniques help you understand data structure, distributions, missing values, and relationships between variables." 
                                                />
                                            </div>
                                            <div className="flex items-center">
                                                <Checkbox
                                                    checked={selectedExplorations.length === availableExplorations.length}
                                                    onCheckedChange={toggleSelectAllExplorations}
                                                />
                                                <span className="ml-2 text-xs">Select All</span>
                                            </div>
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            <div>
                                                {trainFile ? (
                                                    <div className="grid grid-cols-2 gap-1">
                                                        {availableExplorations.map((technique) => (
                                                            <div key={technique} className="flex items-center text-xs">
                                                                <Checkbox
                                                                    checked={selectedExplorations.includes(technique)}
                                                                    onCheckedChange={() => toggleExploration(technique)}
                                                                />
                                                                <span className="ml-1">{technique}</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <div className="text-center">Please select a train file to enable data exploration selection.</div>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm flex items-center">
                                                Select Graphs
                                                <InfoTooltip 
                                                    title="Select Graphs" 
                                                    description="Choose visualizations to generate after training. Graphs help evaluate model performance (ROC, Confusion Matrix), understand features (Feature Importance), and visualize decision boundaries." 
                                                />
                                            </div>
                                            <div className="flex items-center">
                                                <Checkbox
                                                    checked={selectedGraphs.length === availableGraphs.length}
                                                    onCheckedChange={toggleSelectAllGraphs}
                                                />
                                                <span className="ml-2 text-xs">Select All</span>
                                            </div>
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            <div>
                                                {trainFile ? (
                                                    <div className="grid grid-cols-2 gap-1">
                                                        {availableGraphs.map((graph) => (
                                                            <div key={graph} className="flex items-center text-xs">
                                                                <Checkbox
                                                                    checked={selectedGraphs.includes(graph)}
                                                                    onCheckedChange={() => toggleGraph(graph)}
                                                                />
                                                                <span className="ml-1 text-[10px]">{graph}</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <div className="text-center">Please select a train file to enable graph selection.</div>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm flex items-center">
                                                Handling Missing Values
                                                <InfoTooltip 
                                                    title="Handling Missing Values" 
                                                    description="Choose how to handle missing data: Drop rows removes incomplete data, Mean/Median/Mode fills missing values with statistical measures, Forward/Backward Fill uses nearby values." 
                                                />
                                            </div>

                                            {/* Remove Duplicates Checkbox (Only shows if a file is selected) */}
                                            {trainFile && (
                                                <div className="flex items-center text-xs cursor-pointer">
                                                    <Checkbox
                                                        checked={removeDuplicates}
                                                        onCheckedChange={(checked) => setRemoveDuplicates(!!checked)}
                                                        className="mr-2"
                                                    />
                                                    <span>Remove Duplicates</span>
                                                </div>
                                            )}
                                        </div>

                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <div className="grid grid-cols-1 gap-1">
                                                    {availableHandlingMissingValues.map((method) => (
                                                        <label key={method} className="flex items-center text-xs cursor-pointer">
                                                            <input
                                                                type="radio"
                                                                name="missingValueHandling"
                                                                value={method}
                                                                checked={selectedHandlingMissingValue === method}
                                                                onChange={() => setSelectedHandlingMissingValue(method)}
                                                                className="mr-2"
                                                            />
                                                            {method}
                                                        </label>
                                                    ))}
                                                </div>
                                            ) : (
                                                <div className="text-center">Please select a train file to enable missing value handling.</div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Third Row - Classification Options */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-1 mt-1 flex items-center">
                                            Model Configuration
                                            <InfoTooltip 
                                                title="Model Configuration" 
                                                description="Configure the logistic regression model: Encoding converts categorical data to numbers, Solver is the optimization algorithm, Penalty adds regularization to prevent overfitting." 
                                            />
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <>
                                                    <div className="text-[10px] text-gray-400 mb-1">Encoding:</div>
                                                    <Select onValueChange={setEncodingMethod} value={encodingMethod}>
                                                        <SelectTrigger className="w-full text-xs text-white">
                                                            <SelectValue placeholder="Encoding" />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            <SelectItem value="one-hot">One-Hot</SelectItem>
                                                            <SelectItem value="label">Label</SelectItem>
                                                            <SelectItem value="target">Target</SelectItem>
                                                        </SelectContent>
                                                    </Select>

                                                    <div className="text-[10px] text-gray-400 mb-1 mt-2">Solver:</div>
                                                    <Select onValueChange={setSolver} value={solver}>
                                                        <SelectTrigger className="w-full text-xs text-white">
                                                            <SelectValue placeholder="Solver" />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            <SelectItem value="lbfgs">LBFGS</SelectItem>
                                                            <SelectItem value="liblinear">Liblinear</SelectItem>
                                                            <SelectItem value="saga">SAGA</SelectItem>
                                                            <SelectItem value="sag">SAG</SelectItem>
                                                            <SelectItem value="newton-cg">Newton-CG</SelectItem>
                                                        </SelectContent>
                                                    </Select>

                                                    <div className="text-[10px] text-gray-400 mb-1 mt-2">Penalty & C Value:</div>
                                                    <div className="flex gap-2">
                                                        <Select onValueChange={setPenalty} value={penalty}>
                                                            <SelectTrigger className="w-2/3 text-xs text-white">
                                                                <SelectValue placeholder="Penalty" />
                                                            </SelectTrigger>
                                                            <SelectContent>
                                                                <SelectItem value="none">None</SelectItem>
                                                                <SelectItem value="l1">L1</SelectItem>
                                                                <SelectItem value="l2">L2</SelectItem>
                                                                <SelectItem value="elasticnet">ElasticNet</SelectItem>
                                                            </SelectContent>
                                                        </Select>
                                                        <Input 
                                                            type="number" 
                                                            placeholder="C"
                                                            value={cValue}
                                                            onChange={(e) => setCValue(e.target.value)}
                                                            className="w-1/3 text-xs"
                                                            step="0.1"
                                                            min="0.001"
                                                        />
                                                    </div>

                                                    {penalty === "elasticnet" && (
                                                        <div className="mt-2">
                                                            <span className="text-[10px] text-gray-400">L1 Ratio:</span>
                                                            <Input 
                                                                type="number" 
                                                                placeholder="0.5"
                                                                value={l1Ratio}
                                                                onChange={(e) => setL1Ratio(e.target.value)}
                                                                className="w-full text-xs mt-1"
                                                                step="0.1"
                                                                min="0"
                                                                max="1"
                                                            />
                                                        </div>
                                                    )}
                                                </>
                                            ) : (
                                                <div className="text-center text-white text-xs">
                                                    Please select a train file first.
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm flex items-center">
                                                Class Imbalance
                                                <InfoTooltip 
                                                    title="Class Imbalance" 
                                                    description="Handle imbalanced datasets where one class has many more samples than others. SMOTE creates synthetic samples, Over/Under sampling adjusts class sizes, Class weights give more importance to minority classes." 
                                                />
                                            </div>
                                            {trainFile && (
                                                <div className="flex items-center text-xs cursor-pointer">
                                                    <Checkbox
                                                        checked={enableImbalance}
                                                        onCheckedChange={(checked) => setEnableImbalance(!!checked)}
                                                        className="mr-2"
                                                    />
                                                    <span>Enable</span>
                                                </div>
                                            )}
                                        </div>

                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <>
                                                    {enableImbalance ? (
                                                        <>
                                                            <div className="text-[10px] text-gray-400 mb-1">Resampling Method:</div>
                                                            <Select onValueChange={setImbalanceMethod} value={imbalanceMethod}>
                                                                <SelectTrigger className="w-full text-xs text-white">
                                                                    <SelectValue placeholder="Method" />
                                                                </SelectTrigger>
                                                                <SelectContent>
                                                                    <SelectItem value="none">None</SelectItem>
                                                                    <SelectItem value="smote">SMOTE</SelectItem>
                                                                    <SelectItem value="random-over">Random Over</SelectItem>
                                                                    <SelectItem value="random-under">Random Under</SelectItem>
                                                                </SelectContent>
                                                            </Select>

                                                            <div className="text-[10px] text-gray-400 mb-1 mt-2">Class Weights:</div>
                                                            <Select onValueChange={setClassWeight} value={classWeight}>
                                                                <SelectTrigger className="w-full text-xs text-white">
                                                                    <SelectValue placeholder="Weights" />
                                                                </SelectTrigger>
                                                                <SelectContent>
                                                                    <SelectItem value="none">None</SelectItem>
                                                                    <SelectItem value="balanced">Balanced</SelectItem>
                                                                </SelectContent>
                                                            </Select>
                                                        </>
                                                    ) : (
                                                        <div className="text-center text-white text-xs">
                                                            Enable to handle imbalanced classes
                                                        </div>
                                                    )}
                                                </>
                                            ) : (
                                                <div className="text-center text-white text-xs">
                                                    Please select a train file first.
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-1 mt-1 flex items-center">
                                            Feature Scaling & Advanced
                                            <InfoTooltip 
                                                title="Feature Scaling & Advanced" 
                                                description="Feature Scaling normalizes data to similar ranges for better model performance. Max Iterations controls training duration. Random Seed ensures reproducible results. Cross-Validation tests model on multiple data splits." 
                                            />
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <>
                                                    <div className="text-[10px] text-gray-400 mb-1">Feature Scaling:</div>
                                                    <Select onValueChange={(value) => setSelectedFeatureScaling(value === "none" ? null : value)} value={selectedFeatureScaling || "none"}>
                                                        <SelectTrigger className="w-full text-xs text-white">
                                                            <SelectValue placeholder="Scaling" />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            <SelectItem value="none">None</SelectItem>
                                                            <SelectItem value="Min-Max Scaling">Min-Max</SelectItem>
                                                            <SelectItem value="Standard Scaling (Z-score Normalization)">Standard</SelectItem>
                                                            <SelectItem value="Robust Scaling">Robust</SelectItem>
                                                        </SelectContent>
                                                    </Select>

                                                    <div className="text-[10px] text-gray-400 mb-1 mt-2">Max Iterations:</div>
                                                    <Input 
                                                        type="number" 
                                                        placeholder="300"
                                                        value={maxIter}
                                                        onChange={(e) => setMaxIter(e.target.value)}
                                                        className="w-full text-xs"
                                                        min="100"
                                                    />

                                                    <div className="text-[10px] text-gray-400 mb-1 mt-2">Random Seed:</div>
                                                    <Input 
                                                        type="number" 
                                                        placeholder="42"
                                                        value={randomSeed}
                                                        onChange={(e) => setRandomSeed(e.target.value)}
                                                        className="w-full text-xs"
                                                    />

                                                    <div className="flex gap-2 items-center mt-2">
                                                        <Checkbox
                                                            checked={enableCV}
                                                            onCheckedChange={(checked) => setEnableCV(!!checked)}
                                                            className="h-3 w-3"
                                                        />
                                                        <span className="text-xs">Cross-Validation</span>
                                                        <Input 
                                                            type="number" 
                                                            placeholder="5"
                                                            value={cvFolds}
                                                            onChange={(e) => setCvFolds(e.target.value)}
                                                            className="w-16 text-xs h-7"
                                                            min="2"
                                                            max="20"
                                                            disabled={!enableCV}
                                                        />
                                                        <span className="text-[10px] text-gray-400">folds</span>
                                                    </div>
                                                </>
                                            ) : (
                                                <div className="text-center text-white text-xs">
                                                    Please select a train file first.
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Fourth Row - Effect Features for Comparative Graphs */}
                                {(selectedGraphs.some(g => ["Individual Effect Plot", "Mean Effect Plot", "Trend Effect Plot"].includes(g))) && (
                                    <div className="flex gap-x-3 mt-3">
                                        <div className="dark:bg-[#212628] rounded-xl w-full bg-white p-2">
                                            <div className="flex items-center justify-between mb-1 mt-1">
                                                <div className="font-semibold text-sm flex items-center">
                                                    Select Features for Effect Plots
                                                    <InfoTooltip 
                                                        title="Select Features for Effect Plots" 
                                                        description="Choose which features to analyze in effect plots. These graphs show how each feature influences the model's predictions by varying feature values while keeping others constant." 
                                                    />
                                                </div>
                                                {trainFile && selectedTrainColumns.length > 0 && (
                                                    <div className="flex items-center">
                                                        <Checkbox
                                                            checked={selectedEffectFeatures.length === selectedTrainColumns.length}
                                                            onCheckedChange={toggleSelectAllEffectFeatures}
                                                        />
                                                        <span className="ml-1 text-xs">Select All</span>
                                                    </div>
                                                )}
                                            </div>
                                            <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] p-3 rounded-xl overflow-auto max-h-32">
                                                {trainFile && selectedTrainColumns.length > 0 ? (
                                                    <div className="grid grid-cols-6 gap-1">
                                                        {selectedTrainColumns.map((col, index) => (
                                                            <div key={index} className="flex items-center text-xs">
                                                                <Checkbox
                                                                    checked={selectedEffectFeatures.includes(col)}
                                                                    onCheckedChange={() => toggleEffectFeature(col)}
                                                                />
                                                                <span className="ml-1 text-[10px]">{col}</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <div className="text-center text-xs">Please select train columns first.</div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        <TabsContent value="graphs">
                            <div className="border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4 max-h-[700px] overflow-y-auto">
                                {generatedGraphs.length > 0 ? (
                                    <div className="grid grid-cols-3 gap-4">
                                        {generatedGraphs.map((graphPath, index) => {
                                            const graphName = graphPath.split(/[\/\\]/).pop()?.replace('.png', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                            const imageUrl = `/api/users/graphs?path=${encodeURIComponent(graphPath)}`;
                                            return (
                                                <div key={index} className="dark:bg-[#212628] bg-white rounded-lg p-3 shadow-lg relative group">
                                                    <h3 className="text-sm font-semibold mb-2 text-center">{graphName}</h3>
                                                    <img 
                                                        src={imageUrl} 
                                                        alt={graphName || `Graph ${index + 1}`}
                                                        className="w-full h-auto rounded border border-gray-300 dark:border-gray-600 cursor-pointer hover:opacity-90 transition"
                                                        onClick={() => setZoomedGraph(imageUrl)}
                                                        onError={(e) => {
                                                            const target = e.target as HTMLImageElement;
                                                            target.style.display = 'none';
                                                            const parent = target.parentElement;
                                                            if (parent) {
                                                                const errorDiv = document.createElement('div');
                                                                errorDiv.className = 'flex flex-col items-center justify-center p-6 bg-red-50 dark:bg-red-900/20 border-2 border-red-300 dark:border-red-700 rounded-lg text-center min-h-[200px]';
                                                                errorDiv.innerHTML = `
                                                                    <div class="text-4xl mb-2">🚫</div>
                                                                    <div class="text-sm font-semibold text-red-600 dark:text-red-400 mb-2">File Not Found</div>
                                                                    <div class="text-xs text-gray-600 dark:text-gray-400">The graph file has been deleted or moved</div>
                                                                    <div class="text-xs text-gray-500 dark:text-gray-500 mt-2 break-all max-w-full">${graphPath.split(/[\/\\]/).pop()}</div>
                                                                `;
                                                                parent.appendChild(errorDiv);
                                                            }
                                                        }}
                                                    />
                                                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-2">
                                                        <button
                                                            onClick={() => setZoomedGraph(imageUrl)}
                                                            className="bg-blue-600 hover:bg-blue-700 text-white px-2 py-1 rounded text-xs shadow-lg"
                                                            title="Zoom"
                                                        >
                                                            🔍
                                                        </button>
                                                        <button
                                                            onClick={async () => {
                                                                const response = await fetch(imageUrl);
                                                                const blob = await response.blob();
                                                                const url = URL.createObjectURL(blob);
                                                                const a = document.createElement('a');
                                                                a.href = url;
                                                                a.download = graphPath.split(/[\/\\]/).pop() || 'graph.png';
                                                                document.body.appendChild(a);
                                                                a.click();
                                                                document.body.removeChild(a);
                                                                URL.revokeObjectURL(url);
                                                            }}
                                                            className="bg-green-600 hover:bg-green-700 text-white px-2 py-1 rounded text-xs shadow-lg"
                                                            title="Download"
                                                        >
                                                            📥
                                                        </button>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-[650px] text-center space-y-4">
                                        <h2 className="text-4xl font-bold">No Graphs Generated Yet</h2>
                                        <p className="text-xl text-gray-500">
                                            {datasetPath && trainFile
                                                ? `Graphs will be saved in: ${datasetPath.replace(/[\/\\]+$/, "")}/logistic-${trainFile.split(".")[0]}`
                                                : "Run the script to generate and view graphs here"}
                                        </p>
                                    </div>
                                )}
                            </div>

                            {/* Zoom Modal */}
                            {zoomedGraph && (
                                <div 
                                    className="fixed inset-0 bg-black bg-opacity-80 z-50 flex items-center justify-center p-4"
                                    onClick={() => setZoomedGraph(null)}
                                >
                                    <div className="relative max-w-7xl max-h-[90vh] overflow-auto">
                                        <button
                                            onClick={() => setZoomedGraph(null)}
                                            className="absolute top-4 right-4 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-lg font-bold shadow-lg z-10"
                                        >
                                            ✕ Close
                                        </button>
                                        <img 
                                            src={zoomedGraph} 
                                            alt="Zoomed Graph"
                                            className="max-w-full h-auto rounded-lg shadow-2xl"
                                            onClick={(e) => e.stopPropagation()}
                                        />
                                    </div>
                                </div>
                            )}
                        </TabsContent>

                        <TabsContent value="result">
                            <div className="flex flex-col items-center justify-center min-h-[700px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-8">
                                {results ? (
                                    <div className="w-full space-y-6">
                                        <div className="text-center mb-8">
                                            <h2 className="text-4xl font-bold mb-2 bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
                                                📊 Classification Performance Metrics
                                            </h2>
                                            <p className="text-sm text-gray-500 dark:text-gray-400">Generated on {new Date().toLocaleString()}</p>
                                        </div>
                                        
                                        {/* Check if results contain CV table */}
                                        {results.includes("COMPREHENSIVE RESULTS TABLE") ? (
                                            <div className="w-full space-y-6">
                                                {/* Parse and display the CV table */}
                                                {(() => {
                                                    const lines = results.split("\n");
                                                    const hasCV = results.includes("COMPREHENSIVE RESULTS TABLE");
                                                    
                                                    // Extract table rows
                                                    const tableLines = lines.filter(line => {
                                                        const trimmed = line.trim();
                                                        return (trimmed.startsWith("CV Fold") || trimmed.startsWith("Final Model")) && 
                                                               !line.match(/^-+$/);
                                                    });
                                                    
                                                    // Extract metrics for main cards
                                                    const metricsObj: any = {};
                                                    lines.forEach(line => {
                                                        const cleanLine = line.trim();
                                                        if (cleanLine.includes("Accuracy:")) {
                                                            metricsObj.accuracy = cleanLine.split(":")[1]?.trim();
                                                        }
                                                        if (cleanLine.includes("Precision:")) {
                                                            metricsObj.precision = cleanLine.split(":")[1]?.trim();
                                                        }
                                                        if (cleanLine.includes("Recall:")) {
                                                            metricsObj.recall = cleanLine.split(":")[1]?.trim();
                                                        }
                                                        if (cleanLine.includes("F1-Score:")) {
                                                            metricsObj.f1 = cleanLine.split(":")[1]?.trim();
                                                        }
                                                        if (cleanLine.includes("ROC-AUC:")) {
                                                            metricsObj.rocauc = cleanLine.split(":")[1]?.trim();
                                                        }
                                                    });
                                                    
                                                    return (
                                                        <>
                                                            {/* CV Models Table */}
                                                            {hasCV && tableLines.length > 0 && (
                                                                <div className="dark:bg-[#1a1a1a] bg-white rounded-2xl shadow-2xl overflow-hidden border border-blue-500/30 mb-6">
                                                                    <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6">
                                                                        <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                                                                            <span className="text-3xl">🏆</span>
                                                                            All Models Comparison
                                                                        </h3>
                                                                    </div>
                                                                    
                                                                    <div className="overflow-x-auto">
                                                                        <table className="w-full">
                                                                            <thead className="bg-gradient-to-r from-gray-700 to-gray-800 dark:from-gray-800 dark:to-gray-900">
                                                                                <tr>
                                                                                    <th className="px-4 py-4 text-left text-xs font-bold text-white uppercase tracking-wider">Model</th>
                                                                                    <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Accuracy</th>
                                                                                    <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Precision</th>
                                                                                    <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Recall</th>
                                                                                    <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">F1-Score</th>
                                                                                    <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Train Size</th>
                                                                                </tr>
                                                                            </thead>
                                                                            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                                                                                {tableLines.map((row, idx) => {
                                                                                    const cleanRow = row.replace(/-+/g, '').trim();
                                                                                    const parts = cleanRow.split(/\s+/).filter(p => p.length > 0);
                                                                                    const isFinal = row.includes("Final Model");
                                                                                    const rowBg = isFinal 
                                                                                        ? "bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20" 
                                                                                        : idx % 2 === 0 
                                                                                            ? "bg-gray-50 dark:bg-gray-800/50" 
                                                                                            : "bg-white dark:bg-gray-900/50";
                                                                                    
                                                                                    let modelName = '';
                                                                                    let metricsStart = 0;
                                                                                    
                                                                                    if (isFinal) {
                                                                                        modelName = parts.slice(0, 3).join(' ');
                                                                                        metricsStart = 3;
                                                                                    } else {
                                                                                        modelName = parts.slice(0, 3).join(' ');
                                                                                        metricsStart = 3;
                                                                                    }
                                                                                    
                                                                                    return (
                                                                                        <tr key={idx} className={`${rowBg} hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors`}>
                                                                                            <td className="px-4 py-4 whitespace-nowrap">
                                                                                                <div className="flex items-center gap-2">
                                                                                                    {isFinal && <span className="text-xl">⭐</span>}
                                                                                                    <span className={`font-semibold text-sm ${isFinal ? 'text-green-600 dark:text-green-400' : 'text-gray-900 dark:text-gray-100'}`}>
                                                                                                        {modelName}
                                                                                                    </span>
                                                                                                </div>
                                                                                            </td>
                                                                                            <td className="px-3 py-4 text-center">
                                                                                                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                                                                                                    {parts[metricsStart]?.trim() || 'N/A'}
                                                                                                </span>
                                                                                            </td>
                                                                                            <td className="px-3 py-4 text-center">
                                                                                                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                                                                                                    {parts[metricsStart + 1]?.trim() || 'N/A'}
                                                                                                </span>
                                                                                            </td>
                                                                                            <td className="px-3 py-4 text-center">
                                                                                                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                                                                                                    {parts[metricsStart + 2]?.trim() || 'N/A'}
                                                                                                </span>
                                                                                            </td>
                                                                                            <td className="px-3 py-4 text-center">
                                                                                                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                                                                                                    {parts[metricsStart + 3]?.trim() || 'N/A'}
                                                                                                </span>
                                                                                            </td>
                                                                                            <td className="px-3 py-4 text-center">
                                                                                                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200">
                                                                                                    {parts[metricsStart + 4]?.trim() || 'N/A'}
                                                                                                </span>
                                                                                            </td>
                                                                                        </tr>
                                                                                    );
                                                                                })}
                                                                            </tbody>
                                                                        </table>
                                                                    </div>
                                                                </div>
                                                            )}
                                                            
                                                            {/* Main Metrics Cards */}
                                                            <div className="grid grid-cols-5 gap-4 mb-6">
                                                                {metricsObj.accuracy && (
                                                            <div className="dark:bg-gradient-to-br dark:from-purple-900/40 dark:to-purple-700/40 bg-gradient-to-br from-purple-100 to-purple-200 rounded-2xl p-6 shadow-xl border border-purple-500/30">
                                                                <div className="text-center">
                                                                    <div className="text-4xl mb-2">🎯</div>
                                                                    <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">Accuracy</div>
                                                                    <div className="text-3xl font-bold text-purple-600 dark:text-purple-300">{metricsObj.accuracy}</div>
                                                                </div>
                                                            </div>
                                                        )}
                                                        {metricsObj.precision && (
                                                            <div className="dark:bg-gradient-to-br dark:from-blue-900/40 dark:to-blue-700/40 bg-gradient-to-br from-blue-100 to-blue-200 rounded-2xl p-6 shadow-xl border border-blue-500/30">
                                                                <div className="text-center">
                                                                    <div className="text-4xl mb-2">🔍</div>
                                                                    <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">Precision</div>
                                                                    <div className="text-3xl font-bold text-blue-600 dark:text-blue-300">{metricsObj.precision}</div>
                                                                </div>
                                                            </div>
                                                        )}
                                                        {metricsObj.recall && (
                                                            <div className="dark:bg-gradient-to-br dark:from-green-900/40 dark:to-green-700/40 bg-gradient-to-br from-green-100 to-green-200 rounded-2xl p-6 shadow-xl border border-green-500/30">
                                                                <div className="text-center">
                                                                    <div className="text-4xl mb-2">📡</div>
                                                                    <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">Recall</div>
                                                                    <div className="text-3xl font-bold text-green-600 dark:text-green-300">{metricsObj.recall}</div>
                                                                </div>
                                                            </div>
                                                        )}
                                                        {metricsObj.f1 && (
                                                            <div className="dark:bg-gradient-to-br dark:from-yellow-900/40 dark:to-yellow-700/40 bg-gradient-to-br from-yellow-100 to-yellow-200 rounded-2xl p-6 shadow-xl border border-yellow-500/30">
                                                                <div className="text-center">
                                                                    <div className="text-4xl mb-2">⚖️</div>
                                                                    <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">F1-Score</div>
                                                                    <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-300">{metricsObj.f1}</div>
                                                                </div>
                                                            </div>
                                                        )}
                                                        {metricsObj.rocauc && (
                                                            <div className="dark:bg-gradient-to-br dark:from-red-900/40 dark:to-red-700/40 bg-gradient-to-br from-red-100 to-red-200 rounded-2xl p-6 shadow-xl border border-red-500/30">
                                                                <div className="text-center">
                                                                    <div className="text-4xl mb-2">📈</div>
                                                                    <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">ROC-AUC</div>
                                                                    <div className="text-3xl font-bold text-red-600 dark:text-red-300">{metricsObj.rocauc}</div>
                                                                </div>
                                                            </div>
                                                                )}
                                                            </div>
                                                            
                                                            {/* Full Results in Expandable Section */}
                                                            <div className="dark:bg-[#1a1a1a] bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-500/30">
                                                                <div className="bg-gradient-to-r from-green-600 to-blue-600 p-6">
                                                                    <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                                                                        <span className="text-3xl">📋</span>
                                                                        Complete Classification Report
                                                                    </h3>
                                                                </div>
                                                                <div className="p-6">
                                                                    <pre className="whitespace-pre-wrap font-mono text-sm dark:text-gray-300 text-gray-700 overflow-x-auto">
                                                                        {results}
                                                                    </pre>
                                                                </div>
                                                            </div>
                                                        </>
                                                    );
                                                })()}
                                            </div>
                                        ) : (
                                            /* Non-CV Results - Original metric cards */
                                            <div className="w-full space-y-6">
                                                {(() => {
                                                    const lines = results.split("\n");
                                                    const metricsObj: any = {};
                                                    
                                                    lines.forEach(line => {
                                                        const cleanLine = line.trim();
                                                        if (cleanLine.includes("Accuracy:")) {
                                                            metricsObj.accuracy = cleanLine.split(":")[1]?.trim();
                                                        }
                                                        if (cleanLine.includes("Precision:")) {
                                                            metricsObj.precision = cleanLine.split(":")[1]?.trim();
                                                        }
                                                        if (cleanLine.includes("Recall:")) {
                                                            metricsObj.recall = cleanLine.split(":")[1]?.trim();
                                                        }
                                                        if (cleanLine.includes("F1-Score:")) {
                                                            metricsObj.f1 = cleanLine.split(":")[1]?.trim();
                                                        }
                                                        if (cleanLine.includes("ROC-AUC:")) {
                                                            metricsObj.rocauc = cleanLine.split(":")[1]?.trim();
                                                        }
                                                    });
                                                    
                                                    return (
                                                        <>
                                                            {/* Main Metrics Cards */}
                                                            <div className="grid grid-cols-5 gap-4 mb-6">
                                                                {metricsObj.accuracy && (
                                                                    <div className="dark:bg-gradient-to-br dark:from-purple-900/40 dark:to-purple-700/40 bg-gradient-to-br from-purple-100 to-purple-200 rounded-2xl p-6 shadow-xl border border-purple-500/30">
                                                                        <div className="text-center">
                                                                            <div className="text-4xl mb-2">🎯</div>
                                                                            <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">Accuracy</div>
                                                                            <div className="text-3xl font-bold text-purple-600 dark:text-purple-300">{metricsObj.accuracy}</div>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                                {metricsObj.precision && (
                                                                    <div className="dark:bg-gradient-to-br dark:from-blue-900/40 dark:to-blue-700/40 bg-gradient-to-br from-blue-100 to-blue-200 rounded-2xl p-6 shadow-xl border border-blue-500/30">
                                                                        <div className="text-center">
                                                                            <div className="text-4xl mb-2">🔍</div>
                                                                            <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">Precision</div>
                                                                            <div className="text-3xl font-bold text-blue-600 dark:text-blue-300">{metricsObj.precision}</div>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                                {metricsObj.recall && (
                                                                    <div className="dark:bg-gradient-to-br dark:from-green-900/40 dark:to-green-700/40 bg-gradient-to-br from-green-100 to-green-200 rounded-2xl p-6 shadow-xl border border-green-500/30">
                                                                        <div className="text-center">
                                                                            <div className="text-4xl mb-2">📡</div>
                                                                            <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">Recall</div>
                                                                            <div className="text-3xl font-bold text-green-600 dark:text-green-300">{metricsObj.recall}</div>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                                {metricsObj.f1 && (
                                                                    <div className="dark:bg-gradient-to-br dark:from-yellow-900/40 dark:to-yellow-700/40 bg-gradient-to-br from-yellow-100 to-yellow-200 rounded-2xl p-6 shadow-xl border border-yellow-500/30">
                                                                        <div className="text-center">
                                                                            <div className="text-4xl mb-2">⚖️</div>
                                                                            <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">F1-Score</div>
                                                                            <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-300">{metricsObj.f1}</div>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                                {metricsObj.rocauc && (
                                                                    <div className="dark:bg-gradient-to-br dark:from-red-900/40 dark:to-red-700/40 bg-gradient-to-br from-red-100 to-red-200 rounded-2xl p-6 shadow-xl border border-red-500/30">
                                                                        <div className="text-center">
                                                                            <div className="text-4xl mb-2">📈</div>
                                                                            <div className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2">ROC-AUC</div>
                                                                            <div className="text-3xl font-bold text-red-600 dark:text-red-300">{metricsObj.rocauc}</div>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                            
                                                            {/* Full Results */}
                                                            <div className="dark:bg-[#1a1a1a] bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-500/30">
                                                                <div className="bg-gradient-to-r from-green-600 to-blue-600 p-6">
                                                                    <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                                                                        <span className="text-3xl">📋</span>
                                                                        Complete Classification Report
                                                                    </h3>
                                                                </div>
                                                                <div className="p-6">
                                                                    <pre className="whitespace-pre-wrap font-mono text-sm dark:text-gray-300 text-gray-700 overflow-x-auto">
                                                                        {results}
                                                                    </pre>
                                                                </div>
                                                            </div>
                                                        </>
                                                    );
                                                })()}
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                                        <h2 className="text-4xl font-bold mb-4">📋 Classification Results</h2>
                                        <p className="text-xl text-gray-500">Performance metrics will appear here after training</p>
                                        <p className="text-gray-500 mt-2">Run the training script to see detailed results</p>
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        <TabsContent value="terminal">
                            <div className="ml-4 mr-4">
                                <div className="border border-[rgb(61,68,77)] h-[640px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl text-sm p-4 overflow-y-auto">
                                    {logs ? (
                                        <pre ref={terminalRef} className="whitespace-pre-wrap font-mono text-green-400 bg-black p-4 rounded">
                                            {logs}
                                        </pre>
                                    ) : (
                                        <div className="flex flex-col items-center justify-center h-full text-center">
                                            <h2 className="text-2xl font-bold mb-4">🖥️ Terminal Output</h2>
                                            <p className="text-gray-500">Training logs and script output will appear here</p>
                                            <p className="text-gray-500 mt-2">Click the Play button to run the training script</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="predict">
                            <div className="border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-6">
                                {modelTrained ? (
                                    <div className="space-y-6">
                                        <div className="text-center mb-6">
                                            <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                                                🔮 Make Predictions
                                            </h2>
                                            <p className="text-sm text-gray-500 dark:text-gray-400">
                                                Enter feature values to classify {selectedOutputColumn || 'the target'}
                                            </p>
                                        </div>

                                        {/* Model Selector */}
                                        {availableModels.length >= 1 && (
                                            <div className="dark:bg-[#212628] bg-white p-4 rounded-lg border border-blue-500/30">
                                                <Label className="text-sm font-semibold mb-2 block">Select Model for Prediction:</Label>
                                                <Select onValueChange={setSelectedModel} value={selectedModel}>
                                                    <SelectTrigger className="w-full">
                                                        <SelectValue placeholder="Choose a model" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        {availableModels.map((model) => (
                                                            <SelectItem key={model.filename} value={model.filename}>
                                                                {model.name} - Accuracy: {model.accuracy !== null ? model.accuracy.toFixed(4) : 'N/A'}
                                                            </SelectItem>
                                                        ))}
                                                    </SelectContent>
                                                </Select>
                                                <p className="text-xs text-gray-500 mt-2">
                                                    {availableModels.length} {availableModels.length === 1 ? 'model' : 'models'} available
                                                </p>
                                            </div>
                                        )}

                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-[400px] overflow-y-auto p-4 dark:bg-[#212628] bg-white rounded-lg">
                                            {selectedTrainColumns.map((column) => {
                                                const isCategorical = categoricalInfo?.categorical_cols.includes(column);
                                                const isNumeric = categoricalInfo?.numeric_cols.includes(column);
                                                const options = isCategorical ? categoricalInfo?.categorical_values[column] : null;

                                                return (
                                                    <div key={column} className="space-y-2">
                                                        <Label className="text-sm font-semibold">
                                                            {column}
                                                            {isCategorical && <span className="ml-2 text-xs text-blue-500 font-medium">(categorical)</span>}
                                                            {isNumeric && <span className="ml-2 text-xs text-purple-500 font-medium">(numeric)</span>}
                                                        </Label>
                                                        
                                                        {isCategorical && options ? (
                                                            <select
                                                                value={predictionInputs[column] || ""}
                                                                onChange={(e) => 
                                                                    setPredictionInputs(prev => ({
                                                                        ...prev,
                                                                        [column]: e.target.value
                                                                    }))
                                                                }
                                                                className="w-full px-3 py-2 border rounded-lg dark:bg-[#0F0F0F] dark:border-gray-700 dark:text-white focus:ring-2 focus:ring-blue-500"
                                                            >
                                                                <option value="">Select {column}...</option>
                                                                {options.map((option) => (
                                                                    <option key={option} value={option}>
                                                                        {option}
                                                                    </option>
                                                                ))}
                                                            </select>
                                                        ) : (
                                                            <Input
                                                                type="number"
                                                                step="any"
                                                                placeholder={`e.g., 25 (numeric value for ${column})`}
                                                                value={predictionInputs[column] || ""}
                                                                onChange={(e) => 
                                                                    setPredictionInputs(prev => ({
                                                                        ...prev,
                                                                        [column]: e.target.value
                                                                    }))
                                                                }
                                                                className="dark:bg-[#0F0F0F]"
                                                            />
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>

                                        <div className="flex justify-center gap-4">
                                            <Button
                                                onClick={async () => {
                                                    // Check if all fields are filled
                                                    const missingFields = selectedTrainColumns.filter(
                                                        col => !predictionInputs[col] || predictionInputs[col].trim() === ""
                                                    );
                                                    
                                                    if (missingFields.length > 0) {
                                                        alert(`Please fill in all fields. Missing: ${missingFields.join(", ")}`);
                                                        return;
                                                    }

                                                    setIsPredicting(true);
                                                    setPredictionResult(null);

                                                    try {
                                                        // Construct the path to the selected model
                                                        const normalizedPath = datasetPath.trim().replace(/[\/\\]+$/, "");
                                                        const isWindows = navigator.platform.startsWith("Win");
                                                        const separator = isWindows ? "\\\\" : "/";
                                                        const modelDir = `${normalizedPath}${separator}logistic-${trainFile?.split(".")[0]}`;
                                                        const modelPath = `${modelDir}${separator}${selectedModel}`;

                                                        // Prepare input values
                                                        const inputValues: Record<string, any> = {};
                                                        selectedTrainColumns.forEach(col => {
                                                            inputValues[col] = predictionInputs[col];
                                                        });

                                                        const response = await fetch('/api/users/scripts/predict', {
                                                            method: 'POST',
                                                            headers: { 'Content-Type': 'application/json' },
                                                            body: JSON.stringify({
                                                                model_path: modelPath,
                                                                input_values: inputValues
                                                            })
                                                        });

                                                        const data = await response.json();
                                                        
                                                        if (data.error) {
                                                            // Check if it's a retrain required error
                                                            if (data.error.includes('RETRAIN_REQUIRED') || data.error.includes('final_feature_names') || data.error.includes('older version')) {
                                                                setPredictionResult('RETRAIN_REQUIRED');
                                                            } else {
                                                                setPredictionResult(`Error: ${data.error}`);
                                                            }
                                                            setPredictionIsBinary(null);
                                                        } else {
                                                            setPredictionResult(data.prediction);
                                                            setPredictionIsBinary(data.is_binary ?? true);
                                                        }
                                                    } catch (error) {
                                                        setPredictionResult(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
                                                    } finally {
                                                        setIsPredicting(false);
                                                    }
                                                }}
                                                disabled={isPredicting}
                                                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-3 rounded-lg font-semibold"
                                            >
                                                {isPredicting ? (
                                                    <>
                                                        <FaSpinner className="animate-spin mr-2 inline" />
                                                        Predicting...
                                                    </>
                                                ) : (
                                                    "🔮 Predict"
                                                )}
                                            </Button>
                                            
                                            <Button
                                                onClick={() => {
                                                    setPredictionInputs({});
                                                    setPredictionResult(null);
                                                    setPredictionIsBinary(null);
                                                }}
                                                variant="outline"
                                                className="px-8 py-3 rounded-lg font-semibold"
                                            >
                                                🔄 Clear
                                            </Button>
                                        </div>

                                        {predictionResult !== null && (
                                            <div className="mt-6">
                                                {predictionResult === 'RETRAIN_REQUIRED' ? (
                                                    <div className="bg-gradient-to-br from-orange-500 to-red-500 p-8 rounded-2xl shadow-2xl text-white text-center">
                                                        <div className="text-6xl mb-4">⚠️</div>
                                                        <div className="text-2xl font-bold mb-3">
                                                            Model Retrain Required
                                                        </div>
                                                        <div className="text-base mb-4 opacity-90">
                                                            This model was trained with an older version of the system.
                                                        </div>
                                                        <div className="bg-white/20 p-4 rounded-lg text-sm">
                                                            <p className="font-semibold mb-2">To use predictions:</p>
                                                            <ol className="text-left list-decimal list-inside space-y-1">
                                                                <li>Go to the <strong>Home</strong> tab</li>
                                                                <li>Click <strong>▶️ Run</strong> to retrain the model</li>
                                                                <li>Wait for training to complete</li>
                                                                <li>Come back to <strong>Predict</strong> tab</li>
                                                            </ol>
                                                        </div>
                                                        <Button
                                                            onClick={() => {
                                                                setPredictionResult(null);
                                                                // Optionally switch to home tab
                                                            }}
                                                            className="mt-4 bg-white text-orange-600 hover:bg-gray-100"
                                                        >
                                                            Got it!
                                                        </Button>
                                                    </div>
                                                ) : (
                                                    <div className="bg-gradient-to-br from-blue-500 to-purple-500 p-8 rounded-2xl shadow-2xl text-white text-center transform hover:scale-105 transition-transform duration-200">
                                                        <div className="text-6xl mb-4">🎯</div>
                                                        <div className="text-xl font-medium mb-3 opacity-90">
                                                            Predicted Class for {selectedOutputColumn}
                                                        </div>
                                                        <div className="text-6xl font-bold mb-4">
                                                            {predictionResult}
                                                        </div>
                                                        
                                                        {(() => {
                                                        const value = parseFloat(predictionResult);
                                                        if (!isNaN(value)) {
                                                            if (predictionIsBinary === true) {
                                                                const percentage = (value * 100).toFixed(1);
                                                                const category = value >= 0.5 ? "Positive" : "Negative";
                                                                const emoji = value >= 0.5 ? "✅" : "❌";
                                                                return (
                                                                    <div className="mt-4 space-y-2">
                                                                        <div className="text-2xl font-semibold">
                                                                            {emoji} {category} ({percentage}%)
                                                                        </div>
                                                                        <div className="text-sm opacity-75">
                                                                            Confidence: {value >= 0.7 || value <= 0.3 ? "High" : "Moderate"}
                                                                        </div>
                                                                        <div className="text-xs opacity-60 mt-1">
                                                                            Binary Classification Result
                                                                        </div>
                                                                    </div>
                                                                );
                                                            }
                                                        }
                                                        return (
                                                            <div className="text-sm opacity-75 mt-2">
                                                                Classification Result
                                                            </div>
                                                        );
                                                    })()}
                                                    
                                                        <div className="text-sm opacity-75 mt-4">
                                                            Based on the trained model
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        <div className="mt-6 space-y-3">
                                            <div className="p-4 bg-blue-100 dark:bg-blue-900 border-l-4 border-blue-500 rounded-lg">
                                                <p className="text-sm text-blue-800 dark:text-blue-200">
                                                    💡 <strong>Data Entry Guide:</strong>
                                                </p>
                                                <ul className="text-xs text-blue-700 dark:text-blue-300 mt-2 space-y-1 ml-4">
                                                    <li>🔹 <span className="font-semibold">Categorical</span> (dropdown): Select from training values</li>
                                                    <li>🔹 <span className="font-semibold">Numeric</span> (text box): Enter numbers (integers or decimals)</li>
                                                    <li>🔹 Examples: Age → 45, BMI → 28.5, Income → 50000</li>
                                                </ul>
                                            </div>
                                            
                                            <div className="p-4 bg-purple-100 dark:bg-purple-900 border-l-4 border-purple-500 rounded-lg">
                                                <p className="text-sm text-purple-800 dark:text-purple-200">
                                                    📊 <strong>Classification Output:</strong>
                                                </p>
                                                <ul className="text-xs text-purple-700 dark:text-purple-300 mt-2 ml-4 space-y-1">
                                                    <li>• <strong>Binary (0-1):</strong> Shows class prediction with confidence percentage</li>
                                                    <li>• <strong>Multi-class:</strong> Shows the predicted class label</li>
                                                    <li>• The model outputs the most likely class based on input features</li>
                                                </ul>
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-[600px] text-center space-y-4">
                                        <div className="text-8xl mb-4">🔒</div>
                                        <h3 className="text-4xl font-bold bg-gradient-to-r from-gray-600 to-gray-800 dark:from-gray-300 dark:to-gray-500 bg-clip-text text-transparent">
                                            Model Not Trained Yet
                                        </h3>
                                        <p className="text-xl text-gray-500 dark:text-gray-400 max-w-md">
                                            Train your model first to unlock the prediction interface
                                        </p>
                                    </div>
                                )}
                            </div>
                        </TabsContent>
                    </div>
                </Tabs>
            </div>
        </div>
    );
};

export default LogisticRegressionComponent;
