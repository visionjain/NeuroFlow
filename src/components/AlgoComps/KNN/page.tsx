"use client";

import React, { useState, useEffect, useRef } from "react";
import { FaPlay, FaSpinner } from "react-icons/fa";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

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
    
    // Dataset & File Management
    const [trainFile, setTrainFile] = useState<string | null>(null);
    const [testFile, setTestFile] = useState<string | null>(null);
    const [datasetPath, setDatasetPath] = useState<string>("");
    const [showTestUpload, setShowTestUpload] = useState(true);
    const [testSplitRatio, setTestSplitRatio] = useState<string>("0.2");
    
    // Column Selection
    const [trainColumns, setTrainColumns] = useState<string[]>([]);
    const [selectedTrainColumns, setSelectedTrainColumns] = useState<string[]>([]);
    const [selectedOutputColumn, setSelectedOutputColumn] = useState<string | null>(null);
    
    // KNN Configuration
    const [taskType, setTaskType] = useState<string>("classification");
    const [kValue, setKValue] = useState<string>("5");
    const [enableAutoK, setEnableAutoK] = useState(false);
    const [kRangeStart, setKRangeStart] = useState<string>("3");
    const [kRangeEnd, setKRangeEnd] = useState<string>("15");
    const [distanceMetric, setDistanceMetric] = useState<string>("euclidean");
    const [weights, setWeights] = useState<string>("uniform");
    const [algorithm, setAlgorithm] = useState<string>("auto");
    const [leafSize, setLeafSize] = useState<string>("30");
    const [pValue, setPValue] = useState<string>("2");
    
    // Preprocessing
    const [selectedHandlingMissingValue, setSelectedHandlingMissingValue] = useState<string>("drop");
    const [removeDuplicates, setRemoveDuplicates] = useState(true);
    const [encodingMethod, setEncodingMethod] = useState("onehot");
    const [selectedFeatureScaling, setSelectedFeatureScaling] = useState<string>("standard");
    
    // Outlier Detection
    const [enableOutlierDetection, setEnableOutlierDetection] = useState(false);
    const [outlierMethod, setOutlierMethod] = useState("");
    const [zScoreThreshold, setZScoreThreshold] = useState(3.0);
    
    // Advanced Options
    const [enableCV, setEnableCV] = useState(false);
    const [cvFolds, setCvFolds] = useState("5");
    const [enableDimReduction, setEnableDimReduction] = useState(false);
    const [dimReductionMethod, setDimReductionMethod] = useState("pca");
    const [nComponents, setNComponents] = useState("2");
    const [enableImbalance, setEnableImbalance] = useState(false);
    
    // Data Exploration
    const [selectedExplorations, setSelectedExplorations] = useState<string[]>([]);
    
    // Graph Selection
    const [selectedGraphs, setSelectedGraphs] = useState<string[]>([]);
    
    // Effect Features for comparison plots
    const [selectedEffectFeatures, setSelectedEffectFeatures] = useState<string[]>([]);
    
    // UI State
    const [isRunning, setIsRunning] = useState<boolean>(false);
    const [logs, setLogs] = useState<string>("");
    const [results, setResults] = useState<string>("");
    const [generatedGraphs, setGeneratedGraphs] = useState<string[]>([]);
    const [modelTrained, setModelTrained] = useState<boolean>(false);
    const [missingFilesWarning, setMissingFilesWarning] = useState<string[]>([]);
    const [zoomedGraph, setZoomedGraph] = useState<string | null>(null);
    
    // Prediction State
    const [isPredicting, setIsPredicting] = useState<boolean>(false);
    const [predictionInputs, setPredictionInputs] = useState<Record<string, string>>({});
    const [predictionResult, setPredictionResult] = useState<string | null>(null);
    const [predictionIsBinary, setPredictionIsBinary] = useState<boolean | null>(null);
    const [categoricalInfo, setCategoricalInfo] = useState<{
        categorical_cols: string[];
        numeric_cols: string[];
        categorical_values: Record<string, string[]>;
    } | null>(null);
    const [availableModels, setAvailableModels] = useState<any[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>("model.pkl");
    
    const terminalRef = useRef<HTMLDivElement>(null);
    const trainInputRef = useRef<HTMLInputElement>(null);
    const testInputRef = useRef<HTMLInputElement>(null);
    
    const availableExplorations = [
        "First 5 Rows",
        "Last 5 Rows",
        "Dataset Shape",
        "Data Types",
        "Summary Statistics",
        "Missing Values",
        "Unique Values Per Column",
        "Duplicate Rows",
        "Correlation Matrix",
        "Target Distribution"
    ];
    
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
                    setEnableOutlierDetection(state.enableOutlierDetection ?? false);
                    setOutlierMethod(state.outlierMethod || "");
                    setZScoreThreshold(state.zScoreThreshold ?? 3.0);
                    setEncodingMethod(state.encodingMethod || "onehot");
                    setSelectedFeatureScaling(state.selectedFeatureScaling || "standard");
                    setKValue(state.kValue || "5");
                    setEnableAutoK(state.enableAutoK ?? false);
                    setKRangeStart(state.kRangeStart || "1");
                    setKRangeEnd(state.kRangeEnd || "20");
                    setDistanceMetric(state.distanceMetric || "euclidean");
                    setWeights(state.weights || "uniform");
                    setAlgorithm(state.algorithm || "auto");
                    setLeafSize(state.leafSize || "30");
                    setPValue(state.pValue || "2");
                    setEnableCV(state.enableCV ?? false);
                    setCvFolds(state.cvFolds || "5");
                    setEnableDimReduction(state.enableDimReduction ?? false);
                    setDimReductionMethod(state.dimReductionMethod || "pca");
                    setNComponents(state.nComponents || "");
                    setEnableImbalance(state.enableImbalance ?? false);
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
    
    // File handling functions
    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>, type: string) => {
        const file = event.target.files?.[0];
        if (file) {
            const fileName = file.name;
            if (type === "Train") {
                setTrainFile(fileName);
                const reader = new FileReader();
                reader.onload = (e) => {
                    const result = e.target?.result as string;
                    if (result) {
                        try {
                            const firstLine = result.split("\n")[0];
                            // Remove quotes from column names if present
                            const columns = firstLine.split(",").map(col => 
                                col.trim().replace(/^["']|["']$/g, '')
                            );
                            if (columns.length === 0) {
                                throw new Error("No columns found in file");
                            }
                            setTrainColumns(columns);
                        } catch (error) {
                            toast.error("Failed to parse CSV file. Please check the file format.", {
                                duration: 5000,
                            });
                            console.error("CSV parse error:", error);
                        }
                    }
                };
                reader.onerror = () => {
                    toast.error("Failed to read file. File may be corrupted.", {
                        duration: 5000,
                    });
                };
                reader.readAsText(file);
            } else if (type === "Test") {
                setTestFile(fileName);
            }
        }
    };
    
    const toggleTrainColumn = (column: string) => {
        setSelectedTrainColumns(prev =>
            prev.includes(column) ? prev.filter(col => col !== column) : [...prev, column]
        );
    };
    
    const handleOutputColumnSelect = (column: string) => {
        setSelectedOutputColumn(column);
    };
    
    const toggleSelectAll = () => {
        if (selectedTrainColumns.length === trainColumns.length) {
            setSelectedTrainColumns([]);
        } else {
            setSelectedTrainColumns([...trainColumns]);
        }
    };
    
    const toggleTestDataset = () => {
        setShowTestUpload(prev => {
            setTrainColumns([]);
            setSelectedTrainColumns([]);
            setSelectedOutputColumn(null);
            setTrainFile(null);
            setTestFile(null);
            if (trainInputRef.current) trainInputRef.current.value = "";
            if (testInputRef.current) testInputRef.current.value = "";
            return !prev;
        });
    };
    
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
        }
    };
    
    // Data Exploration handlers
    const toggleExploration = (technique: string) => {
        setSelectedExplorations(prev =>
            prev.includes(technique) ? prev.filter(item => item !== technique) : [...prev, technique]
        );
    };
    
    const toggleSelectAllExplorations = () => {
        if (selectedExplorations.length === availableExplorations.length) {
            setSelectedExplorations([]);
        } else {
            setSelectedExplorations(availableExplorations);
        }
    };
    
    // Graph selection handlers
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
    
    // Effect features handlers
    const toggleEffectFeature = (feature: string) => {
        setSelectedEffectFeatures((prev) =>
            prev.includes(feature) ? prev.filter((item) => item !== feature) : [...prev, feature]
        );
    };

    const toggleSelectAllEffectFeatures = () => {
        if (selectedEffectFeatures.length === selectedTrainColumns.length) {
            setSelectedEffectFeatures([]);
        } else {
            setSelectedEffectFeatures([...selectedTrainColumns]);
        }
    };
    
    // Check if any effect-based graph is selected
    const hasEffectGraphsSelected = selectedGraphs.some(graph => 
        ["Decision Boundary", "Neighbor Analysis", "Feature Impact"].includes(graph)
    );
    
    // Dynamically generate available graphs based on CV settings
    const getAvailableGraphs = () => {
        // Add learning curve options at the top
        const learningCurves = ["Learning Curve - Overall"];
        if (enableCV) {
            learningCurves.push("Learning Curve - All Folds");
        }
        
        const knnGraphs = [
            "K vs Accuracy",
            "Distance Distribution", 
            "Decision Boundary",
            "Neighbor Analysis"
        ];
        
        const classificationGraphs = [
            "Confusion Matrix",
            "ROC Curve",
            "Precision-Recall"
        ];
        
        const regressionGraphs = [
            "Residual Plot",
            "Predicted vs Actual",
            "Error Distribution"
        ];
        
        const featureGraphs = [
            "Correlation Heatmap",
            "PCA Visualization",
            "Box Plots"
        ];
        
        return [...learningCurves, ...knnGraphs, ...classificationGraphs, ...regressionGraphs, ...featureGraphs];
    };
    
    const availableGraphs = getAvailableGraphs();
    
    // Terminal auto-scroll
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
                    const modelDir = `${normalizedPath}${separator}knn-${trainFile?.split(".")[0]}`;
                    const modelPath = `${modelDir}${separator}model.pkl`;

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
                        setSelectedModel("model.pkl");
                        
                        // Save state again now that we have available models (silent save)
                        setTimeout(() => {
                            saveProjectState(undefined, true);
                        }, 500);
                    }
                } catch (error) {
                    console.error("Failed to fetch categorical info:", error);
                    toast.error("Failed to load model information. Files may be missing or corrupted.", {
                        duration: 5000,
                    });
                }
            };

            fetchCategoricalInfo();
        }
    }, [modelTrained, datasetPath, trainFile]);
    
    // KNN Config handlers
    const handleKValueChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        if (value === "" || /^\d+$/.test(value)) {
            setKValue(value);
        }
    };
    
    const handleKRangeStartChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        if (value === "" || /^\d+$/.test(value)) {
            setKRangeStart(value);
        }
    };
    
    const handleKRangeEndChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        if (value === "" || /^\d+$/.test(value)) {
            setKRangeEnd(value);
        }
    };
    
    const handleLeafSizeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        if (value === "" || /^\d+$/.test(value)) {
            setLeafSize(value);
        }
    };
    
    const handlePValueChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        if (value === "" || /^\d+$/.test(value)) {
            setPValue(value);
        }
    };
    
    const handleCVFoldsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        if (value === "" || /^\d+$/.test(value)) {
            const num = parseInt(value);
            if (num >= 2 && num <= 10) {
                setCvFolds(value);
            } else if (value === "") {
                setCvFolds("");
            }
        }
    };
    
    const handleNComponentsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        if (value === "" || /^\d+$/.test(value)) {
            setNComponents(value);
        }
    };
    
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
                enableOutlierDetection,
                outlierMethod,
                zScoreThreshold,
                encodingMethod,
                selectedFeatureScaling,
                kValue,
                enableAutoK,
                kRangeStart,
                kRangeEnd,
                distanceMetric,
                weights,
                algorithm,
                leafSize,
                pValue,
                enableCV,
                cvFolds,
                enableDimReduction,
                dimReductionMethod,
                nComponents,
                enableImbalance,
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
        setLogs(""); // Clear previous logs
        setResults(""); // Clear previous results
        setGeneratedGraphs([]); // Clear previous graphs
        setModelTrained(false); // Reset model trained status
        setPredictionResult(null); // Clear previous predictions
        setIsRunning(true); // Disable button

        // Validation checks
        if (!datasetPath || !trainFile) {
            toast.error("Dataset path and train file are required");
            setIsRunning(false);
            return;
        }

        if (selectedTrainColumns.length === 0) {
            toast.error("Please select at least one train column");
            setIsRunning(false);
            return;
        }

        if (!selectedOutputColumn) {
            toast.error("Please select an output column");
            setIsRunning(false);
            return;
        }

        if (!selectedFeatureScaling) {
            toast.error("Feature scaling is required for KNN (distance-based algorithm)");
            setIsRunning(false);
            return;
        }

        // Validate K value or auto-K range
        if (!enableAutoK) {
            const k = parseInt(kValue);
            if (!k || k < 1) {
                toast.error("K value must be a positive integer");
                setIsRunning(false);
                return;
            }
        } else {
            const start = parseInt(kRangeStart);
            const end = parseInt(kRangeEnd);
            if (!start || !end || start >= end || start < 1) {
                toast.error("Invalid K range. Start must be less than End and both must be positive");
                setIsRunning(false);
                return;
            }
        }

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
            feature_scaling: selectedFeatureScaling,
            available_Explorations: JSON.stringify(selectedExplorations),

            // KNN specific parameters
            k_value: kValue || "5",
            enable_auto_k: JSON.stringify(enableAutoK),
            k_range_start: kRangeStart || "1",
            k_range_end: kRangeEnd || "20",
            distance_metric: distanceMetric,
            weights: weights,
            algorithm: algorithm,
            leaf_size: leafSize || "30",
            
            // Minkowski specific
            ...(distanceMetric === "minkowski" && {
                p_value: pValue || "2"
            }),

            // Outlier Detection Parameters
            enable_outlier_detection: JSON.stringify(enableOutlierDetection),
            outlier_method: JSON.stringify(outlierMethod),
            ...(enableOutlierDetection && outlierMethod === "zscore" && {
                z_score_threshold: JSON.stringify(zScoreThreshold)
            }),

            // Advanced options
            enable_cv: JSON.stringify(enableCV),
            cv_folds: cvFolds || "5",
            enable_dim_reduction: JSON.stringify(enableDimReduction),
            dim_reduction_method: dimReductionMethod,
            n_components: nComponents || "auto",
            enable_imbalance: JSON.stringify(enableImbalance),
            
            // Effect features for comparison
            effect_features: JSON.stringify(selectedEffectFeatures.length > 0 ? selectedEffectFeatures : [selectedTrainColumns[0]]),
        });

        if (!testFile && testSplitRatio) {
            queryParams.append("test_split_ratio", testSplitRatio);
        }

        const apiUrl = `/api/users/scripts/knn?${queryParams.toString()}`;

        // Local variable to accumulate all output lines
        let allLogs = "";
        let trainingSuccessful = false;
        const eventSource = new EventSource(apiUrl);

        eventSource.onmessage = (event) => {
            if (event.data === "END_OF_STREAM") {
                trainingSuccessful = allLogs.includes("FINISHED SUCCESSFULLY");
                
                // Extract results section - include both evaluation and comprehensive results table
                let resultsText = "";
                const hasComprehensiveTable = allLogs.includes("COMPREHENSIVE RESULTS TABLE");
                const hasEvaluation = allLogs.includes("MODEL EVALUATION");
                
                if (hasComprehensiveTable) {
                    // If CV was used, extract from MODEL EVALUATION through TRAINING COMPLETE
                    const lines = allLogs.split("\n");
                    const evalStartIdx = lines.findIndex(line => line.includes("MODEL EVALUATION"));
                    const completeIdx = lines.findIndex(line => line.includes("TRAINING COMPLETE"));
                    
                    if (evalStartIdx !== -1 && completeIdx !== -1) {
                        // Include MODEL EVALUATION section and COMPREHENSIVE RESULTS TABLE
                        resultsText = lines.slice(evalStartIdx, completeIdx).join("\n");
                    } else if (evalStartIdx !== -1) {
                        resultsText = lines.slice(evalStartIdx).join("\n");
                    }
                } else if (hasEvaluation) {
                    // No CV, just extract MODEL EVALUATION section
                    const lines = allLogs.split("\n");
                    const evalStartIdx = lines.findIndex(line => line.includes("MODEL EVALUATION"));
                    const graphsIdx = lines.findIndex(line => line.includes("GENERATING GRAPHS"));
                    
                    if (evalStartIdx !== -1) {
                        const endIdx = graphsIdx !== -1 ? graphsIdx : lines.length;
                        resultsText = lines.slice(evalStartIdx, endIdx).join("\n");
                    }
                }
                
                // Fallback for KNN metrics
                if (!resultsText) {
                    const resultLines = allLogs
                        .split("\n")
                        .filter(line =>
                            line.startsWith("Accuracy:") ||
                            line.startsWith("Precision:") ||
                            line.startsWith("Recall:") ||
                            line.startsWith("F1-Score:") ||
                            line.startsWith("Mean Squared Error:") ||
                            line.startsWith("R-squared Score:") ||
                            line.startsWith("Optimal K:") ||
                            line.includes("Best K value")
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
                
                if (trainingSuccessful) {
                    setModelTrained(true);
                    toast.success("KNN model trained successfully!", {
                        style: {
                            background: 'green',
                            color: 'white',
                        },
                    });
                    
                    // Save project state
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
                        enableOutlierDetection,
                        outlierMethod,
                        zScoreThreshold,
                        encodingMethod,
                        selectedFeatureScaling,
                        kValue,
                        enableAutoK,
                        kRangeStart,
                        kRangeEnd,
                        distanceMetric,
                        weights,
                        algorithm,
                        leafSize,
                        pValue,
                        enableCV,
                        cvFolds,
                        enableDimReduction,
                        dimReductionMethod,
                        nComponents,
                        enableImbalance,
                        selectedGraphs,
                        selectedExplorations,
                        logs: allLogs,
                        results: resultsText,
                        generatedGraphs: parsedGraphs,
                        modelTrained: true,
                        availableModels
                    });
                } else {
                    setModelTrained(false);
                    setLogs((prev) => prev + "\n❌ Training failed. Prediction tab remains locked.\n");
                    toast.error("Training failed. Check terminal for details.");
                }
            } else {
                // Filter out the JSON marker from terminal display
                if (!event.data.includes("__GENERATED_GRAPHS_JSON__")) {
                    allLogs += event.data + "\n";
                    setLogs((prev) => prev + event.data + "\n");
                } else {
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
            toast.error("Connection error. Check if backend server is running.");
        };
    };

    return (
        <div>
            <div className="text-xl">
                <Tabs defaultValue="home">
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

                        <TabsList className="flex w-[50%] text-black dark:text-white bg-[#e6e6e6] dark:bg-[#0F0F0F]">
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]" value="home">
                                Home
                            </TabsTrigger>
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]" value="graphs">
                                Graphs
                            </TabsTrigger>
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]" value="result">
                                Results
                            </TabsTrigger>
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]" value="terminal">
                                Terminal
                            </TabsTrigger>
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628] disabled:opacity-50 disabled:cursor-not-allowed" value="predict" disabled={!modelTrained}>
                                {modelTrained ? "🔮 Predict" : "🔒 Predict"}
                            </TabsTrigger>
                        </TabsList>

                        <div className="flex gap-2">
                            <Button className="rounded-xl" onClick={handleRunScript} disabled={isRunning}>
                                {isRunning ? <FaSpinner className="animate-spin" /> : <FaPlay />}
                            </Button>
                            
                            {modelTrained ? (
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
                                                    setSelectedEffectFeatures([]);
                                                    setCategoricalInfo(null);
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
                            ) : null}
                        </div>
                    </div>

                    <div className="mt-2">
                        <TabsContent value="home">
                            <div className="border border-[rgb(61,68,77)] flex flex-col gap-3 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4">
                                
                                {/* Dataset Compatibility Info */}
                                <div className="dark:bg-[#1a1d1f] bg-[#f5f5f5] rounded-xl p-4 border-2 border-blue-500 dark:border-blue-600">
                                    <h3 className="text-lg font-bold mb-2 text-center">📋 K-Nearest Neighbors - Supported Dataset Types</h3>
                                    <div className="grid grid-cols-3 gap-4 text-sm">
                                        <div>
                                            <p className="font-semibold text-blue-600 dark:text-blue-400">✅ Classification:</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Binary/Multi-class categories (Iris, Wine, Digits)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Examples: Disease diagnosis, Customer segmentation</p>
                                        </div>
                                        <div>
                                            <p className="font-semibold text-green-600 dark:text-green-400">✅ Regression:</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Continuous numeric prediction (House prices, Stock)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Examples: Real estate, Sales forecast</p>
                                        </div>
                                        <div>
                                            <p className="font-semibold text-red-600 dark:text-red-400">⚠️ Important:</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Feature Scaling: MANDATORY (distance-based)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Works best with &lt;20 features</p>
                                        </div>
                                    </div>
                                </div>

                                {/* First Row - Dataset Upload */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4">
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
                                        <div className="flex w-full gap-2">
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
                                                    className="h-12 w-full border-2 border-dashed border-gray-500 rounded-md"
                                                    onClick={() => trainInputRef.current?.click()}
                                                >
                                                    {trainFile ? (
                                                        <span className="text-sm truncate w-full text-center">{trainFile}</span>
                                                    ) : (
                                                        <span className="text-3xl">+</span>
                                                    )}
                                                </Button>
                                            </div>
                                            {showTestUpload ? (
                                                <div className="flex flex-col w-full items-center">
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
                                                        className="h-12 w-full border-2 border-dashed border-gray-500 rounded-md"
                                                        onClick={() => testInputRef.current?.click()}
                                                    >
                                                        {testFile ? (
                                                            <span className="text-sm truncate w-full text-center">{testFile}</span>
                                                        ) : (
                                                            <span className="text-3xl">+</span>
                                                        )}
                                                    </Button>
                                                </div>
                                            ) : (
                                                <div className="flex flex-col w-full items-center">
                                                    <Label className="text-sm font-semibold mb-1">Test Split Ratio</Label>
                                                    <Input
                                                        type="text"
                                                        value={testSplitRatio}
                                                        onChange={handleTestSplitChange}
                                                        onBlur={handleTestSplitBlur}
                                                        placeholder="0.2"
                                                        className="w-full h-12 text-center dark:bg-[#0F0F0F]"
                                                    />
                                                </div>
                                            )}
                                        </div>
                                        <p
                                            className="underline mt-2 flex justify-center text-sm text-blue-600 cursor-pointer"
                                            onClick={toggleTestDataset}
                                        >
                                            {showTestUpload ? "Don't have test dataset?" : "Have a test dataset?"}
                                        </p>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm flex items-center">
                                                Select Train Columns
                                                <InfoTooltip 
                                                    title="Train Columns" 
                                                    description="Choose which columns from your dataset to use as input features for training the KNN model. These are the independent variables (X) that will be used to predict the target based on nearest neighbors." 
                                                />
                                            </div>
                                            {trainFile && (
                                                <div className="flex items-center">
                                                    <Checkbox
                                                        checked={selectedTrainColumns.length === trainColumns.length}
                                                        onCheckedChange={toggleSelectAll}
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
                                                <div className="text-center text-sm text-gray-500">
                                                    Select train file to enable column selection
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-1 mt-1 flex items-center">
                                            Select Output Column
                                            <InfoTooltip 
                                                title="Output Column" 
                                                description="Select the target variable (dependent variable) that you want to predict. For classification, this should be categorical. For regression, this should be numeric." 
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
                                                <div className="text-center text-sm text-gray-500">
                                                    Select train file to enable output selection
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Second Row - Data Exploration & Preprocessing */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm flex items-center">
                                                Data Exploration
                                                <InfoTooltip 
                                                    title="Data Exploration" 
                                                    description="Select exploratory data analysis techniques to understand your dataset better. Includes statistics, distributions, correlations, and data quality checks before training." 
                                                />
                                            </div>
                                            <div className="flex items-center">
                                                <Checkbox
                                                    checked={selectedExplorations.length === availableExplorations.length}
                                                    onCheckedChange={toggleSelectAllExplorations}
                                                />
                                                <span className="ml-1 text-xs">Select All</span>
                                            </div>
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            <div className="space-y-1 text-xs">
                                                {availableExplorations.map((exploration, idx) => (
                                                    <div key={idx} className="flex items-center">
                                                        <Checkbox
                                                            checked={selectedExplorations.includes(exploration)}
                                                            onCheckedChange={() => toggleExploration(exploration)}
                                                        />
                                                        <span className="ml-1">{exploration}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-1 mt-1 flex items-center">
                                            Data Cleaning
                                            <InfoTooltip 
                                                title="Data Cleaning" 
                                                description="Prepare your data by handling missing values, removing duplicates, and encoding categorical features. KNN requires clean numeric data for distance calculations." 
                                            />
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-2 rounded-xl space-y-2 overflow-auto">
                                            <div>
                                                <Label className="text-xs font-semibold">Missing Values</Label>
                                                <Select value={selectedHandlingMissingValue} onValueChange={setSelectedHandlingMissingValue}>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="drop">Drop Rows</SelectItem>
                                                        <SelectItem value="mean">Mean</SelectItem>
                                                        <SelectItem value="median">Median</SelectItem>
                                                        <SelectItem value="mode">Mode</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <Checkbox
                                                    id="remove-dup"
                                                    checked={removeDuplicates}
                                                    onCheckedChange={(checked) => setRemoveDuplicates(!!checked)}
                                                />
                                                <Label htmlFor="remove-dup" className="text-xs">Remove Duplicates</Label>
                                            </div>
                                            <div>
                                                <Label className="text-xs font-semibold">Encoding</Label>
                                                <Select value={encodingMethod} onValueChange={setEncodingMethod}>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="onehot">One-Hot</SelectItem>
                                                        <SelectItem value="label">Label</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-2 mt-1 flex items-center">
                                            Outlier Detection
                                            <InfoTooltip 
                                                title="Outlier Detection" 
                                                description="Detect and handle outliers that may negatively affect KNN performance. Methods include Z-Score (statistical), IQR (quartile-based), and Winsorization (capping extreme values)." 
                                            />
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl space-y-2">
                                            <div className="flex items-center space-x-2">
                                                <Checkbox
                                                    id="enable-outlier"
                                                    checked={enableOutlierDetection}
                                                    onCheckedChange={(checked) => setEnableOutlierDetection(!!checked)}
                                                />
                                                <Label htmlFor="enable-outlier" className="text-xs">Enable Outlier Detection</Label>
                                            </div>
                                            <div>
                                                <Label className="text-xs font-semibold">Detection Method</Label>
                                                <Select
                                                    value={outlierMethod}
                                                    onValueChange={setOutlierMethod}
                                                    disabled={!enableOutlierDetection}
                                                >
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue placeholder="Select method" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="zscore">Z-Score</SelectItem>
                                                        <SelectItem value="iqr">IQR</SelectItem>
                                                        <SelectItem value="winsor">Winsorization</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            {enableOutlierDetection && outlierMethod === "zscore" && (
                                                <div>
                                                    <Label className="text-xs font-semibold">Z-Score Threshold</Label>
                                                    <Input
                                                        type="number"
                                                        value={zScoreThreshold}
                                                        onChange={(e) => setZScoreThreshold(parseFloat(e.target.value))}
                                                        className="h-8 text-xs dark:bg-[#212628]"
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Third Row - Feature Scaling & KNN Config */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-xs mb-1 mt-1 flex items-center">
                                            <span className="text-red-500 mr-1">*</span>
                                            Feature Scaling
                                            <InfoTooltip 
                                                title="Feature Scaling (REQUIRED)" 
                                                description="MANDATORY for KNN since it's a distance-based algorithm. Scales all features to the same range to prevent features with larger values from dominating the distance calculation. Choose Min-Max (0-1), Standard (mean=0, std=1), or Robust (median-based, handles outliers)." 
                                            />
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl space-y-2">
                                            <div className="bg-red-50 dark:bg-red-900/20 border border-red-300 rounded p-2 text-xs text-center">
                                                ⚠️ KNN requires feature scaling
                                            </div>
                                            <div className="space-y-1 text-xs">
                                                <div className="flex items-center">
                                                    <Checkbox 
                                                        checked={selectedFeatureScaling === "minmax"}
                                                        onCheckedChange={(checked) => checked && setSelectedFeatureScaling("minmax")}
                                                    />
                                                    <span className="ml-1">Min-Max Scaling</span>
                                                </div>
                                                <div className="flex items-center">
                                                    <Checkbox 
                                                        checked={selectedFeatureScaling === "standard"}
                                                        onCheckedChange={(checked) => checked && setSelectedFeatureScaling("standard")}
                                                    />
                                                    <span className="ml-1">Standard Scaling (Recommended)</span>
                                                </div>
                                                <div className="flex items-center">
                                                    <Checkbox 
                                                        checked={selectedFeatureScaling === "robust"}
                                                        onCheckedChange={(checked) => checked && setSelectedFeatureScaling("robust")}
                                                    />
                                                    <span className="ml-1">Robust Scaling</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-2 mt-1 flex items-center">
                                            KNN Algorithm Config
                                            <InfoTooltip 
                                                title="KNN Algorithm Config" 
                                                description="Configure K-Nearest Neighbors parameters: K value (number of neighbors), distance metric (Euclidean, Manhattan, etc.), weighting (uniform or distance-based), and search algorithm (auto, ball tree, kd tree, brute force)." 
                                            />
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl space-y-2 overflow-auto">
                                            <div>
                                                <Label className="text-xs font-semibold">K Value (Neighbors)</Label>
                                                <Input 
                                                    type="number" 
                                                    placeholder="5" 
                                                    value={kValue}
                                                    onChange={handleKValueChange}
                                                    disabled={enableAutoK}
                                                    className="h-8 text-xs dark:bg-[#212628]" 
                                                />
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <Checkbox 
                                                    id="auto-k"
                                                    checked={enableAutoK}
                                                    onCheckedChange={(checked) => setEnableAutoK(!!checked)}
                                                />
                                                <Label htmlFor="auto-k" className="text-xs">Auto-Find Optimal K</Label>
                                            </div>
                                            {enableAutoK && (
                                                <div className="flex gap-2">
                                                    <div className="flex-1">
                                                        <Label className="text-xs">K Start</Label>
                                                        <Input 
                                                            type="number" 
                                                            placeholder="1"
                                                            value={kRangeStart}
                                                            onChange={handleKRangeStartChange}
                                                            className="h-8 text-xs dark:bg-[#212628]" 
                                                        />
                                                    </div>
                                                    <div className="flex-1">
                                                        <Label className="text-xs">K End</Label>
                                                        <Input 
                                                            type="number" 
                                                            placeholder="20"
                                                            value={kRangeEnd}
                                                            onChange={handleKRangeEndChange}
                                                            className="h-8 text-xs dark:bg-[#212628]" 
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                            <div>
                                                <Label className="text-xs font-semibold">Distance Metric</Label>
                                                <Select value={distanceMetric} onValueChange={setDistanceMetric}>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="euclidean">Euclidean</SelectItem>
                                                        <SelectItem value="manhattan">Manhattan</SelectItem>
                                                        <SelectItem value="minkowski">Minkowski</SelectItem>
                                                        <SelectItem value="chebyshev">Chebyshev</SelectItem>
                                                        <SelectItem value="cosine">Cosine</SelectItem>
                                                        <SelectItem value="hamming">Hamming</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            {distanceMetric === "minkowski" && (
                                                <div>
                                                    <Label className="text-xs font-semibold">P Value</Label>
                                                    <Input 
                                                        type="number" 
                                                        placeholder="2"
                                                        value={pValue}
                                                        onChange={handlePValueChange}
                                                        className="h-8 text-xs dark:bg-[#212628]" 
                                                    />
                                                </div>
                                            )}
                                            <div>
                                                <Label className="text-xs font-semibold">Weight Function</Label>
                                                <Select value={weights} onValueChange={setWeights}>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="uniform">Uniform</SelectItem>
                                                        <SelectItem value="distance">Distance</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-2 mt-1 flex items-center">
                                            Advanced Options
                                            <InfoTooltip 
                                                title="Advanced Options" 
                                                description="Enable advanced features: Cross-validation for robust evaluation, PCA for dimensionality reduction, and imbalance handling for skewed datasets. These improve model performance and reliability." 
                                            />
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl space-y-2 overflow-auto">
                                            <div>
                                                <Label className="text-xs font-semibold">Algorithm Type</Label>
                                                <Select value={algorithm} onValueChange={setAlgorithm}>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="auto">Auto</SelectItem>
                                                        <SelectItem value="ball_tree">Ball Tree</SelectItem>
                                                        <SelectItem value="kd_tree">KD Tree</SelectItem>
                                                        <SelectItem value="brute">Brute Force</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            {(algorithm === "ball_tree" || algorithm === "kd_tree") && (
                                                <div>
                                                    <Label className="text-xs font-semibold">Leaf Size</Label>
                                                    <Input 
                                                        type="number" 
                                                        placeholder="30"
                                                        value={leafSize}
                                                        onChange={handleLeafSizeChange}
                                                        className="h-8 text-xs dark:bg-[#212628]" 
                                                    />
                                                </div>
                                            )}
                                            <div className="flex items-center space-x-2">
                                                <Checkbox 
                                                    id="enable-cv"
                                                    checked={enableCV}
                                                    onCheckedChange={(checked) => setEnableCV(!!checked)}
                                                />
                                                <Label htmlFor="enable-cv" className="text-xs">Enable Cross-Validation</Label>
                                            </div>
                                            <div>
                                                <Label className="text-xs font-semibold">CV Folds (2-10)</Label>
                                                <Input 
                                                    type="number" 
                                                    placeholder="5"
                                                    value={cvFolds}
                                                    onChange={handleCVFoldsChange}
                                                    disabled={!enableCV}
                                                    className="h-8 text-xs dark:bg-[#212628]" 
                                                />
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <Checkbox 
                                                    id="dim-reduction"
                                                    checked={enableDimReduction}
                                                    onCheckedChange={(checked) => setEnableDimReduction(!!checked)}
                                                />
                                                <Label htmlFor="dim-reduction" className="text-xs">Dimensionality Reduction (PCA)</Label>
                                            </div>
                                            {enableDimReduction && (
                                                <div>
                                                    <Label className="text-xs font-semibold">PCA Components</Label>
                                                    <Input 
                                                        type="number" 
                                                        placeholder="Auto"
                                                        value={nComponents}
                                                        onChange={handleNComponentsChange}
                                                        className="h-8 text-xs dark:bg-[#212628]" 
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Fourth Row - Graph Selection */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-full bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm flex items-center">
                                                Select Graphs to Generate ({availableGraphs.length} Available)
                                                <InfoTooltip 
                                                    title="Select Graphs" 
                                                    description="Choose visualizations to generate: Learning Curves (training progress), KNN-specific plots (K vs Accuracy, Decision Boundary), classification metrics (Confusion Matrix, ROC), regression analysis (Residual Plot), and feature analysis (PCA, Correlation Heatmap). Enable CV to unlock fold-specific learning curves." 
                                                />
                                            </div>
                                            <div className="flex items-center">
                                                <Checkbox 
                                                    checked={selectedGraphs.length === availableGraphs.length}
                                                    onCheckedChange={toggleSelectAllGraphs}
                                                />
                                                <span className="ml-1 text-xs">Select All</span>
                                            </div>
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            <div className="grid grid-cols-4 gap-2 text-xs">
                                                {availableGraphs.map((graph) => (
                                                    <div key={graph} className="flex items-center">
                                                        <Checkbox 
                                                            checked={selectedGraphs.includes(graph)}
                                                            onCheckedChange={() => toggleGraph(graph)}
                                                        />
                                                        <span className="ml-1">{graph}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Fifth Row - Feature Comparison Selection (Conditional) */}
                                {hasEffectGraphsSelected && trainFile && selectedTrainColumns.length > 0 && (
                                    <div className="flex gap-x-3">
                                        <div className="dark:bg-[#212628] h-52 rounded-xl w-full bg-white p-2">
                                            <div className="flex items-center justify-between mb-1 mt-1">
                                                <div className="font-semibold text-sm flex items-center">
                                                    📊 Select Features for Neighbor Analysis & Decision Boundary
                                                    <InfoTooltip 
                                                        title="Effect Plot Features" 
                                                        description="Choose which features to analyze in neighbor analysis and decision boundary plots. These visualizations show how each feature influences KNN predictions by examining local neighborhoods and decision regions. Useful for understanding feature impact in the KNN algorithm." 
                                                    />
                                                    <span className="text-xs font-normal text-gray-500 ml-2">
                                                        (Choose which features to analyze against {selectedOutputColumn || 'target'})
                                                    </span>
                                                </div>
                                                <div className="flex items-center">
                                                    <Checkbox
                                                        checked={selectedEffectFeatures.length === selectedTrainColumns.length}
                                                        onCheckedChange={toggleSelectAllEffectFeatures}
                                                    />
                                                    <span className="ml-2 text-xs">Select All</span>
                                                </div>
                                            </div>
                                            <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                                <div className="grid grid-cols-4 gap-2">
                                                    {selectedTrainColumns.map((feature) => (
                                                        <div key={feature} className="flex items-center text-xs">
                                                            <Checkbox
                                                                checked={selectedEffectFeatures.includes(feature)}
                                                                onCheckedChange={() => toggleEffectFeature(feature)}
                                                            />
                                                            <span className="ml-1">{feature}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                                {selectedEffectFeatures.length === 0 && (
                                                    <div className="text-center text-yellow-600 dark:text-yellow-400 mt-4 text-xs">
                                                        ⚠️ No features selected. Will use first feature ({selectedTrainColumns[0]}) by default.
                                                    </div>
                                                )}
                                                {selectedEffectFeatures.length > 0 && (
                                                    <div className="text-center text-green-600 dark:text-green-400 mt-4 text-xs">
                                                        ✓ Will generate plots for: {selectedEffectFeatures.join(', ')}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )}

                            </div>
                        </TabsContent>

                        <TabsContent value="terminal">
                            <div className="ml-4 mr-4">
                                <div
                                    ref={terminalRef}
                                    className="border border-[rgb(61,68,77)] h-[640px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl text-sm p-4 overflow-y-auto"
                                >
                                    <pre className="whitespace-pre-wrap">{logs || "Terminal Output will be shown here."}</pre>
                                </div>
                                {logs && !isRunning && (
                                    <div className="mt-4 flex justify-end">
                                        <Button
                                            onClick={() => {
                                                const blob = new Blob([logs], { type: 'text/plain' });
                                                const url = URL.createObjectURL(blob);
                                                const a = document.createElement('a');
                                                a.href = url;
                                                a.download = `${projectName}_logs_${new Date().toISOString().split('T')[0]}.txt`;
                                                document.body.appendChild(a);
                                                a.click();
                                                document.body.removeChild(a);
                                                URL.revokeObjectURL(url);
                                            }}
                                            className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
                                        >
                                            📥 Download Logs
                                        </Button>
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        <TabsContent value="result">
                            <div className="flex flex-col items-center justify-center min-h-[700px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-8">
                                {results ? (
                                    <div className="w-full space-y-6">
                                        <div className="text-center mb-8">
                                            <h2 className="text-4xl font-bold mb-2 bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
                                                📊 KNN Model Performance
                                            </h2>
                                            <p className="text-sm text-gray-500 dark:text-gray-400">Generated on {new Date().toLocaleString()}</p>
                                        </div>

                                        

                                        {/* Extract and display CV fold results table if available */}
                                        {(() => {
                                            const lines = results.split('\n');
                                            const hasCV = results.includes('COMPREHENSIVE RESULTS TABLE');
                                            
                                            // Extract CV fold model rows (exclude separators and header)
                                            const tableLines = lines.filter(line => {
                                                const trimmed = line.trim();
                                                // Include lines that start with "CV Fold" or "Final Model"
                                                // Exclude separator lines (only dashes) and the header line (contains column names)
                                                return (trimmed.startsWith('CV Fold') || trimmed.startsWith('Final Model')) && 
                                                       !line.match(/^[-=\s]+$/) && 
                                                       !trimmed.match(/^Model\s+(Accuracy|R²)/i);
                                            });
                                            
                                            const isClassification = results.includes('Accuracy') && results.includes('Precision');
                                            
                                            if (hasCV && tableLines.length > 0) {
                                                return (
                                                    <div className="dark:bg-[#1a1a1a] bg-white rounded-2xl shadow-2xl overflow-hidden border border-purple-500/30 mb-6 w-full max-w-5xl mx-auto">
                                                        <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6">
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
                                                                        {isClassification ? (
                                                                            <>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Accuracy</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Precision</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Recall</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">F1-Score</th>
                                                                            </>
                                                                        ) : (
                                                                            <>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">R²</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">MSE</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">MAE</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">RMSE</th>
                                                                            </>
                                                                        )}
                                                                    </tr>
                                                                </thead>
                                                                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                                                                    {tableLines.map((line, idx) => {
                                                                        const parts = line.trim().split(/\s{2,}/);
                                                                        const modelName = parts[0];
                                                                        const isFinal = modelName.includes('Final Model');
                                                                        
                                                                        return (
                                                                            <tr key={idx} className={`${
                                                                                isFinal 
                                                                                    ? 'bg-gradient-to-r from-green-500/20 to-blue-500/20 dark:from-green-600/20 dark:to-blue-600/20 font-bold' 
                                                                                    : idx % 2 === 0 
                                                                                        ? 'dark:bg-[#1a1a1a] bg-gray-50' 
                                                                                        : 'dark:bg-[#212628] bg-white'
                                                                            } hover:bg-purple-500/10 transition-colors`}>
                                                                                <td className="px-4 py-4 text-sm font-medium">{modelName}</td>
                                                                                {parts.slice(1).map((val, i) => (
                                                                                    <td key={i} className="px-3 py-4 text-sm text-center font-mono">{val}</td>
                                                                                ))}
                                                                            </tr>
                                                                        );
                                                                    })}
                                                                </tbody>
                                                            </table>
                                                        </div>
                                                    </div>
                                                );
                                            }
                                            return null;
                                        })()}
                                        

                                        {/* Extract and display key metrics if available */}
                                        {(() => {
                                            const lines = results.split('\n');
                                            const metrics: any = {};
                                            
                                            lines.forEach(line => {
                                                // Skip warning lines and only extract actual metrics
                                                if (line.includes('⚠️') || line.includes('Skipped')) return;
                                                
                                                if (line.includes('Accuracy:')) metrics.accuracy = line.split(':')[1]?.trim();
                                                if (line.includes('Precision:')) metrics.precision = line.split(':')[1]?.trim();
                                                if (line.includes('Recall:')) metrics.recall = line.split(':')[1]?.trim();
                                                if (line.includes('F1-Score:') || line.includes('F1 Score:')) metrics.f1 = line.split(':')[1]?.trim();
                                                if (line.includes('R-squared') || line.includes('R²')) metrics.r2 = line.split(':')[1]?.trim();
                                                if (line.includes('MSE:') || line.includes('Mean Squared Error:')) metrics.mse = line.split(':')[1]?.trim();
                                                if (line.includes('RMSE:') || line.includes('Root Mean Squared Error:')) metrics.rmse = line.split(':')[1]?.trim();
                                                if (line.includes('MAE:') || line.includes('Mean Absolute Error:')) metrics.mae = line.split(':')[1]?.trim();
                                                if (line.includes('Optimal K:') || line.includes('Best K:')) metrics.optimalK = line.split(':')[1]?.trim();
                                            });

                                            // Only show metric cards if we found any metrics
                                            if (Object.keys(metrics).length > 0) {
                                                return (
                                                    <div className="w-full max-w-5xl mx-auto mt-6">
                                                        <h3 className="text-2xl font-bold mb-4 text-center">Key Metrics Summary</h3>
                                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                                            {metrics.accuracy && (
                                                                <div className="dark:bg-gradient-to-br from-green-600 to-green-700 bg-gradient-to-br from-green-400 to-green-500 rounded-xl p-4 text-white shadow-lg">
                                                                    <div className="text-sm font-semibold mb-1">Accuracy</div>
                                                                    <div className="text-3xl font-bold">{metrics.accuracy}</div>
                                                                </div>
                                                            )}
                                                            {metrics.precision && (
                                                                <div className="dark:bg-gradient-to-br from-blue-600 to-blue-700 bg-gradient-to-br from-blue-400 to-blue-500 rounded-xl p-4 text-white shadow-lg">
                                                                    <div className="text-sm font-semibold mb-1">Precision</div>
                                                                    <div className="text-3xl font-bold">{metrics.precision}</div>
                                                                </div>
                                                            )}
                                                            {metrics.recall && (
                                                                <div className="dark:bg-gradient-to-br from-purple-600 to-purple-700 bg-gradient-to-br from-purple-400 to-purple-500 rounded-xl p-4 text-white shadow-lg">
                                                                    <div className="text-sm font-semibold mb-1">Recall</div>
                                                                    <div className="text-3xl font-bold">{metrics.recall}</div>
                                                                </div>
                                                            )}
                                                            {metrics.f1 && (
                                                                <div className="dark:bg-gradient-to-br from-orange-600 to-orange-700 bg-gradient-to-br from-orange-400 to-orange-500 rounded-xl p-4 text-white shadow-lg">
                                                                    <div className="text-sm font-semibold mb-1">F1-Score</div>
                                                                    <div className="text-3xl font-bold">{metrics.f1}</div>
                                                                </div>
                                                            )}
                                                            {metrics.r2 && (
                                                                <div className="dark:bg-gradient-to-br from-teal-600 to-teal-700 bg-gradient-to-br from-teal-400 to-teal-500 rounded-xl p-4 text-white shadow-lg">
                                                                    <div className="text-sm font-semibold mb-1">R² Score</div>
                                                                    <div className="text-3xl font-bold">{metrics.r2}</div>
                                                                </div>
                                                            )}
                                                            {metrics.mse && (
                                                                <div className="dark:bg-gradient-to-br from-red-600 to-red-700 bg-gradient-to-br from-red-400 to-red-500 rounded-xl p-4 text-white shadow-lg">
                                                                    <div className="text-sm font-semibold mb-1">MSE</div>
                                                                    <div className="text-3xl font-bold">{metrics.mse}</div>
                                                                </div>
                                                            )}
                                                            {metrics.rmse && (
                                                                <div className="dark:bg-gradient-to-br from-pink-600 to-pink-700 bg-gradient-to-br from-pink-400 to-pink-500 rounded-xl p-4 text-white shadow-lg">
                                                                    <div className="text-sm font-semibold mb-1">RMSE</div>
                                                                    <div className="text-3xl font-bold">{metrics.rmse}</div>
                                                                </div>
                                                            )}
                                                            {metrics.mae && (
                                                                <div className="dark:bg-gradient-to-br from-indigo-600 to-indigo-700 bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-xl p-4 text-white shadow-lg">
                                                                    <div className="text-sm font-semibold mb-1">MAE</div>
                                                                    <div className="text-3xl font-bold">{metrics.mae}</div>
                                                                </div>
                                                            )}
                                                            {metrics.optimalK && (
                                                                <div className="dark:bg-gradient-to-br from-yellow-600 to-yellow-700 bg-gradient-to-br from-yellow-400 to-yellow-500 rounded-xl p-4 text-white shadow-lg">
                                                                    <div className="text-sm font-semibold mb-1">Optimal K</div>
                                                                    <div className="text-3xl font-bold">{metrics.optimalK}</div>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                );
                                            }
                                            return null;
                                        })()}
                                        {/* Display results as formatted text */}
                                        <div className="w-full max-w-5xl mx-auto">
                                            <div className="dark:bg-[#212628] bg-white rounded-xl p-6 shadow-lg border border-gray-300 dark:border-gray-600">
                                                <pre className="whitespace-pre-wrap text-sm font-mono overflow-x-auto">
                                                    {results}
                                                </pre>
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-center space-y-4">
                                        <div className="text-6xl mb-4">📊</div>
                                        <h2 className="text-3xl font-bold">No Results Yet</h2>
                                        <p className="text-xl text-gray-500">
                                            Train your KNN model to see performance metrics here
                                        </p>
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
                                                ? `Graphs will be saved in: ${datasetPath.replace(/[/\\]+$/, "")}/knn-${trainFile.split(".")[0]}`
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

                        <TabsContent value="predict">
                            <div className="border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-6">
                                {modelTrained ? (
                                    <div className="space-y-6">
                                        <div className="text-center mb-6">
                                            <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
                                                🔮 Make Predictions with KNN
                                            </h2>
                                            <p className="text-sm text-gray-500 dark:text-gray-400">
                                                Enter feature values to predict {selectedOutputColumn || 'the target'}
                                            </p>
                                        </div>

                                        {/* Model Selector */}
                                        {availableModels.length >= 1 ? (
                                            <div className="dark:bg-[#212628] bg-white p-4 rounded-lg border border-green-500/30">
                                                <Label className="text-sm font-semibold mb-2 block">Select Model for Prediction:</Label>
                                                <Select onValueChange={setSelectedModel} value={selectedModel}>
                                                    <SelectTrigger className="w-full dark:bg-[#1a1a1a] bg-white dark:text-white text-black border dark:border-gray-600 border-gray-300">
                                                        <SelectValue>
                                                            {(() => {
                                                                const model = availableModels.find(m => m.filename === selectedModel);
                                                                if (!model) return "Final Model (model.pkl)";
                                                                return (
                                                                    <span>
                                                                        {model.name}
                                                                        {model.accuracy !== null && model.accuracy !== undefined && ` | Acc: ${model.accuracy.toFixed(4)}`}
                                                                        {model.r2_score !== null && model.r2_score !== undefined && ` | R²: ${model.r2_score.toFixed(4)}`}
                                                                    </span>
                                                                );
                                                            })()}
                                                        </SelectValue>
                                                    </SelectTrigger>
                                                    <SelectContent className="dark:bg-[#1a1a1a] bg-white">
                                                        {availableModels.map((model) => (
                                                            <SelectItem key={model.filename} value={model.filename} className="dark:text-white text-black">
                                                                {model.name}
                                                                {model.accuracy !== null && model.accuracy !== undefined && ` | Acc: ${model.accuracy.toFixed(4)}`}
                                                                {model.r2_score !== null && model.r2_score !== undefined && ` | R²: ${model.r2_score.toFixed(4)}`}
                                                            </SelectItem>
                                                        ))}
                                                    </SelectContent>
                                                </Select>
                                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                                                    {availableModels.length} {availableModels.length === 1 ? 'model' : 'models'} available
                                                    {availableModels.find(m => m.filename === selectedModel)?.type === 'cv_fold' && 
                                                        ` | Using CV Fold ${availableModels.find(m => m.filename === selectedModel)?.fold_number}`
                                                    }
                                                </p>
                                            </div>
                                        ) : (
                                            <div className="dark:bg-[#212628] bg-white p-4 rounded-lg border border-gray-500/30">
                                                <Label className="text-sm font-semibold mb-2 block">Select Model for Prediction:</Label>
                                                <div className="w-full dark:bg-[#1a1a1a] bg-gray-100 p-3 rounded border dark:border-gray-600 border-gray-300">
                                                    <span className="dark:text-white text-black">Final Model (model.pkl)</span>
                                                </div>
                                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                                                    Using default model
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
                                                            {isCategorical && <span className="ml-2 text-xs text-green-500 font-medium">(categorical)</span>}
                                                            {isNumeric && <span className="ml-2 text-xs text-blue-500 font-medium">(numeric)</span>}
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
                                                                className="w-full px-3 py-2 border rounded-lg dark:bg-[#0F0F0F] dark:border-gray-700 dark:text-white focus:ring-2 focus:ring-green-500"
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
                                                                placeholder={`Enter ${column}`}
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
                                                        toast.error(`Please fill in all fields. Missing: ${missingFields.join(", ")}`);
                                                        return;
                                                    }

                                                    setIsPredicting(true);
                                                    setPredictionResult(null);

                                                    try {
                                                        // Construct the path to the selected model
                                                        const normalizedPath = datasetPath.trim().replace(/[/\\]+$/, "");
                                                        const isWindows = navigator.platform.startsWith("Win");
                                                        const separator = isWindows ? "\\\\" : "/";
                                                        const modelDir = `${normalizedPath}${separator}knn-${trainFile?.split(".")[0]}`;
                                                        const modelPath = `${modelDir}${separator}${selectedModel}`;

                                                        // Prepare input values as an object with feature names
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
                                                            setPredictionResult(`Error: ${data.error}`);
                                                            setPredictionIsBinary(null);
                                                            toast.error(data.error);
                                                        } else {
                                                            setPredictionResult(data.prediction);
                                                            setPredictionIsBinary(data.is_binary ?? null);
                                                            toast.success("Prediction completed!", {
                                                                style: {
                                                                    background: 'green',
                                                                    color: 'white',
                                                                },
                                                            });
                                                        }
                                                    } catch (error) {
                                                        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
                                                        setPredictionResult(`Error: ${errorMsg}`);
                                                        toast.error(errorMsg);
                                                    } finally {
                                                        setIsPredicting(false);
                                                    }
                                                }}
                                                disabled={isPredicting}
                                                className="bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white px-8 py-3 rounded-lg font-semibold"
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
                                                <div className="bg-gradient-to-br from-green-500 to-blue-500 p-8 rounded-2xl shadow-2xl text-white text-center transform hover:scale-105 transition-transform duration-200">
                                                    <div className="text-6xl mb-4">🎯</div>
                                                    <div className="text-xl font-medium mb-3 opacity-90">
                                                        Predicted {selectedOutputColumn}
                                                    </div>
                                                    <div className="text-6xl font-bold mb-4">
                                                        {predictionResult}
                                                    </div>
                                                    
                                                    {/* Smart interpretation */}
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
                                                            } else if (predictionIsBinary === false) {
                                                                return (
                                                                    <div className="mt-4 space-y-2">
                                                                        <div className="text-sm opacity-75">
                                                                            📈 Regression Prediction
                                                                        </div>
                                                                        <div className="text-xs opacity-60">
                                                                            Continuous value predicted by KNN
                                                                        </div>
                                                                    </div>
                                                                );
                                                            }
                                                        }
                                                        return null;
                                                    })()}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-[600px] text-center space-y-4">
                                        <div className="text-6xl mb-4">🔒</div>
                                        <h2 className="text-3xl font-bold">Prediction Locked</h2>
                                        <p className="text-xl text-gray-500">
                                            Train your KNN model first to unlock predictions
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

export default KNNComponent;
