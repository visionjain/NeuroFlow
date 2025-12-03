"use client";

import React, { useState, useEffect, useRef } from "react";
import { FaPlay, FaSpinner } from "react-icons/fa";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";

interface LinearRegressionProps {
    projectName: string;
    projectAlgo: string;
    projectTime: string;
}

const LinearRegressionComponent: React.FC<LinearRegressionProps> = ({ projectName, projectAlgo, projectTime }) => {
    const [trainFile, setTrainFile] = useState<string | null>(null);
    const [testFile, setTestFile] = useState<string | null>(null);
    const [datasetPath, setDatasetPath] = useState<string>("");
    const [showTestUpload, setShowTestUpload] = useState(true);
    const trainInputRef = useRef<HTMLInputElement>(null);
    const testInputRef = useRef<HTMLInputElement>(null);
    const terminalRef = useRef<HTMLDivElement>(null);
    const [isRunning, setIsRunning] = useState<boolean>(false);
    const [logs, setLogs] = useState<string>("");
    const [testSplitRatio, setTestSplitRatio] = useState<string>("0.2");
    const [trainColumns, setTrainColumns] = useState<string[]>([]);
    const [selectedTrainColumns, setSelectedTrainColumns] = useState<string[]>([]);
    const [selectedOutputColumn, setSelectedOutputColumn] = useState<string | null>(null);
    const [results, setResults] = useState<string>("");
    const [selectedGraphs, setSelectedGraphs] = useState<string[]>([]);
    const [selectedHandlingMissingValue, setSelectedHandlingMissingValue] = useState<string>("Drop Rows with Missing Values");
    const [removeDuplicates, setRemoveDuplicates] = useState(true);
    const [enableOutlierDetection, setEnableOutlierDetection] = useState(false);
    const [outlierMethod, setOutlierMethod] = useState(""); // Selected method
    const [zScoreThreshold, setZScoreThreshold] = useState(3.0);
    const [iqrLower, setIqrLower] = useState(1.5);
    const [iqrUpper, setIqrUpper] = useState(1.5);
    const [winsorLower, setWinsorLower] = useState(1);
    const [winsorUpper, setWinsorUpper] = useState(99);
    const [encodingMethod, setEncodingMethod] = useState("one-hot");
    const [regularizationType, setRegularizationType] = useState("none");
    const [alphaValue, setAlphaValue] = useState("1.0");
    const [enableCV, setEnableCV] = useState(false);
    const [cvFolds, setCvFolds] = useState("5");
    const [selectedExplorations, setSelectedExplorations] = useState<string[]>([]);
    const [selectedFeatureScaling, setSelectedFeatureScaling] = useState<string | null>(null);
    const [generatedGraphs, setGeneratedGraphs] = useState<string[]>([]);
    const [selectedEffectFeatures, setSelectedEffectFeatures] = useState<string[]>([]);
    const [zoomedGraph, setZoomedGraph] = useState<string | null>(null);
    const [modelTrained, setModelTrained] = useState<boolean>(false);
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
    const [selectedModel, setSelectedModel] = useState<string>("model.pkl");

    const availableFeatureScaling = [
        "Min-Max Scaling",
        "Standard Scaling (Z-score Normalization)",
        "Robust Scaling"
    ];


    // State for multi-choice section
    const [selectedFeatureEngineering, setSelectedFeatureEngineering] = useState<string[]>([]);





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
        "Target Column Distribution"
    ];


    const availableGraphs = [
        "Heatmap",
        "Histogram Distribution",
        "Histogram Residuals",
        "Individual Effect Plot",
        "Mean Effect Plot",
        "Model Coefficients",
        "Residual Plot",
        "Shap Summary Plot",
        "Trend Effect Plot",
        "Box Plot",
    ];
    const availableHandlingMissingValues = [
        "Mean Imputation",
        "Median Imputation",
        "Mode Imputation",
        "Forward/Backward Fill",
        "Drop Rows with Missing Values",
    ];


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
                    const modelDir = `${normalizedPath}${separator}linearregression-${trainFile?.split(".")[0]}`;
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
                    }
                } catch (error) {
                    console.error("Failed to fetch categorical info:", error);
                }
            };

            fetchCategoricalInfo();
        }
    }, [modelTrained, datasetPath, trainFile]);



    const handleTestSplitChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        let value = event.target.value;

        // Allow only valid decimal numbers (empty string is also allowed for smooth typing)
        if (/^\d*\.?\d{0,2}$/.test(value) || value === "") {
            setTestSplitRatio(value);
        }
    };
    const handleTestSplitBlur = () => {
        let numValue = parseFloat(testSplitRatio);
        if (isNaN(numValue) || numValue < 0.01 || numValue > 0.99) {
            setTestSplitRatio("0.2"); // Default value
        }
    };


    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>, type: string) => {
        const file = event.target.files?.[0];
        if (file) {
            const fileName = file.name;

            if (type === "Train") {
                setTrainFile(fileName);

                // Read the first line of the CSV to get column names
                const reader = new FileReader();
                reader.onload = (e) => {
                    const result = e.target?.result as string;
                    if (result) {
                        const firstLine = result.split("\n")[0];
                        const columns = firstLine.split(",").map(col => col.trim()); // Handle spaces
                        setTrainColumns(columns);
                    }
                };
                reader.readAsText(file);
            } else if (type === "Test") {
                setTestFile(fileName);
            }
        }
    };

    // Toggle individual selection
    const toggleExploration = (technique: string) => {
        setSelectedExplorations((prev) =>
            prev.includes(technique) ? prev.filter((item) => item !== technique) : [...prev, technique]
        );
    };

    // Select all / Deselect all
    const toggleSelectAllExplorations = () => {
        if (selectedExplorations.length === availableExplorations.length) {
            setSelectedExplorations([]); // Deselect all
        } else {
            setSelectedExplorations(availableExplorations); // Select all
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






    const toggleTestDataset = () => {
        setShowTestUpload((prev) => {
            // Always clear the columns when the toggle is clicked
            setTrainColumns([]); // Clear the train columns
            setSelectedTrainColumns([]); // Reset the selected train columns
            setSelectedOutputColumn(null); // Reset the output column

            // Reset the file selections
            setTrainFile(null);
            setTestFile(null);

            // Reset the input fields for file selection
            if (trainInputRef.current) trainInputRef.current.value = "";
            if (testInputRef.current) testInputRef.current.value = "";

            return !prev;
        });
    };


    const toggleGraph = (graph: string) => {
        setSelectedGraphs((prev) =>
            prev.includes(graph)
                ? prev.filter((g) => g !== graph)
                : [...prev, graph]
        );
    };

    // Function to select/deselect all graphs
    const toggleSelectAllGraphs = () => {
        if (selectedGraphs.length === availableGraphs.length) {
            setSelectedGraphs([]);
        } else {
            setSelectedGraphs([...availableGraphs]);
        }
    };






    const handleRunScript = () => {
        setLogs(""); // Clear previous logs
        setResults(""); // Clear previous results
        setGeneratedGraphs([]); // Clear previous graphs
        setModelTrained(false); // Reset model trained status
        setPredictionResult(null); // Clear previous predictions
        setIsRunning(true); // Disable button

        if (!datasetPath || !trainFile) {
            alert("Dataset path and train file are required.");
            setIsRunning(false);
            return;
        }

        if (selectedTrainColumns.length === 0) {
            alert("Please select at least one train column.");
            setIsRunning(false);
            return;
        }

        if (!selectedOutputColumn) {
            alert("Please select an output column.");
            setIsRunning(false);
            return;
        }

        // Normalize datasetPath to remove trailing slashes
        let normalizedPath = datasetPath.trim();
        if (normalizedPath.endsWith("\\") || normalizedPath.endsWith("/")) {
            normalizedPath = normalizedPath.slice(0, -1);
        }

        // Detect OS to use appropriate separator (this example assumes Windows or Unix-like)
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
            regularization_type: regularizationType,
            alpha: alphaValue,
            enable_cv: JSON.stringify(enableCV),
            cv_folds: cvFolds,
            available_Explorations: JSON.stringify(selectedExplorations),

            // Outlier Detection Parameters
            enable_outlier_detection: JSON.stringify(enableOutlierDetection),
            outlier_method: JSON.stringify(outlierMethod), // "Z-score" / "IQR" / "Winsorization"

            // Z-score Method (if selected)
            ...(outlierMethod === "Z-score" && {
                z_score_threshold: JSON.stringify(zScoreThreshold),
            }),

            // IQR Method (if selected)
            ...(outlierMethod === "IQR" && {
                iqr_lower: JSON.stringify(iqrLower),
                iqr_upper: JSON.stringify(iqrUpper),
            }),

            // Winsorization Method (if selected)
            ...(outlierMethod === "Winsorization" && {
                winsor_lower: JSON.stringify(winsorLower),
                winsor_upper: JSON.stringify(winsorUpper),
            }),

            // Feature Scaling (Single Selection)
            feature_scaling: selectedFeatureScaling ? selectedFeatureScaling : "",

            // Effect Features for Comparison
            effect_features: JSON.stringify(selectedEffectFeatures.length > 0 ? selectedEffectFeatures : [selectedTrainColumns[0]]),

        });



        if (!testFile && testSplitRatio) {
            queryParams.append("test_split_ratio", testSplitRatio);
        }

        const apiUrl = `/api/users/scripts/linearregression?${queryParams.toString()}`;

        // Local variable to accumulate all output lines
        let allLogs = "";
        let trainingSuccessful = false; // Track if training completed successfully
        const eventSource = new EventSource(apiUrl);

        eventSource.onmessage = (event) => {
            if (event.data === "END_OF_STREAM") {
                // Check if we saw "FINISHED SUCCESSFULLY" in the logs
                trainingSuccessful = allLogs.includes("FINISHED SUCCESSFULLY");
                
                // Once stream ends, parse accumulated logs for results
                // Extract the entire MODEL PERFORMANCE SUMMARY or MODEL RESULTS section
                let resultsText = "";
                
                // Try to find the comprehensive table section (CV) or simple results (non-CV)
                const hasCV = allLogs.includes("MODEL PERFORMANCE SUMMARY");
                const hasResults = allLogs.includes("MODEL RESULTS");
                
                if (hasCV || hasResults) {
                    const searchString = hasCV ? "MODEL PERFORMANCE SUMMARY" : "MODEL RESULTS";
                    const startIdx = allLogs.indexOf(searchString);
                    const endIdx = allLogs.indexOf("FINISHED SUCCESSFULLY", startIdx);
                    
                    if (startIdx !== -1) {
                        // Extract everything from the start of the section to either end marker or end of logs
                        const extractedSection = endIdx !== -1 
                            ? allLogs.substring(startIdx, endIdx).trim()
                            : allLogs.substring(startIdx).trim();
                        
                        // Include the separator lines for better formatting
                        const lines = allLogs.split("\n");
                        const summaryLineIdx = lines.findIndex(line => line.includes(searchString));
                        if (summaryLineIdx >= 0) {
                            // Get a few lines before the summary (to include the equals separator)
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
                
                // Fallback to line-by-line extraction if no table found
                if (!resultsText) {
                    const resultLines = allLogs
                        .split("\n")
                        .filter(
                            (line) =>
                                line.startsWith("Mean Squared Error:") ||
                                line.startsWith("Mean Squared Error (MSE):") ||
                                line.startsWith("R-squared Score:") ||
                                line.startsWith("R-squared Score (R²):") ||
                                line.startsWith("Accuracy Score:") ||
                                line.startsWith("CV Mean R² Score:") ||
                                line.startsWith("CV Mean MSE:") ||
                                line.includes("Skipping accuracy") ||
                                line.includes("Regression task")
                        );
                    resultsText = resultLines.join("\n");
                }
                
                setResults(resultsText);
                console.log("Results extracted:", resultsText ? "Success" : "Empty");
                
                // Parse generated graphs JSON
                const graphsMatch = allLogs.match(/__GENERATED_GRAPHS_JSON__(.+?)__END_GRAPHS__/);
                if (graphsMatch) {
                    try {
                        const graphs = JSON.parse(graphsMatch[1]);
                        setGeneratedGraphs(graphs);
                    } catch (e) {
                        console.error("Failed to parse graphs JSON:", e);
                    }
                }
                
                eventSource.close();
                setIsRunning(false);
                
                // ONLY enable prediction tab if training was successful
                if (trainingSuccessful) {
                    setModelTrained(true);
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
            setModelTrained(false); // Keep prediction tab locked on error
            setLogs((prev) => prev + "\n❌ Connection error or training interrupted.\n");
        };
    };



    const toggleSelectAll = () => {
        if (selectedTrainColumns.length === trainColumns.length) {
            setSelectedTrainColumns([]); // deselect all
        } else {
            setSelectedTrainColumns([...trainColumns]); // select all
        }
    };

    // Toggle individual effect feature
    const toggleEffectFeature = (feature: string) => {
        setSelectedEffectFeatures((prev) =>
            prev.includes(feature) ? prev.filter((item) => item !== feature) : [...prev, feature]
        );
    };

    // Select all / Deselect all effect features
    const toggleSelectAllEffectFeatures = () => {
        if (selectedEffectFeatures.length === selectedTrainColumns.length) {
            setSelectedEffectFeatures([]); // Deselect all
        } else {
            setSelectedEffectFeatures([...selectedTrainColumns]); // Select all
        }
    };

    // Check if any effect-based graph is selected
    const hasEffectGraphsSelected = selectedGraphs.some(graph => 
        ["Individual Effect Plot", "Mean Effect Plot", "Trend Effect Plot", "Effect Plot"].includes(graph)
    );









    return (
        <div>
            <div className="text-xl">
                {/* Tabs Wraps Everything Now */}
                <Tabs defaultValue="home">
                    {/* Project Title & Tabs in One Row */}
                    <div className="flex items-center justify-between px-4 mt-2">
                        <div className="font-bold">
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


                        <Button className="rounded-xl" onClick={handleRunScript} disabled={isRunning}>
                            {isRunning ? <FaSpinner className="animate-spin" /> : <FaPlay />}
                        </Button>

                    </div>

                    {/* Tabs Content (Stays Fixed in Place) */}
                    <div className="mt-2">
                        <TabsContent value="home">
                            <div className="border border-[rgb(61,68,77)] flex flex-col gap-3 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4">
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
                                            <div className="font-semibold text-sm">Select Train Columns</div>

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
                                        <div className="font-semibold text-sm mb-1 mt-1">Select Output Column</div>

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
                                            <div className="font-semibold text-sm">Select Data Exploration Techniques</div>
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
                                            <div className="font-semibold text-sm">Select Graphs</div>
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
                                                    <>

                                                        <div className="grid grid-cols-2 gap-1">
                                                            {availableGraphs.map((graph) => (
                                                                <div key={graph} className="flex items-center text-xs">
                                                                    <Checkbox
                                                                        checked={selectedGraphs.includes(graph)}
                                                                        onCheckedChange={() => toggleGraph(graph)}
                                                                    />
                                                                    <span className="ml-1">{graph}</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </>
                                                ) : (
                                                    <div className="text-center">Please select a train file to enable graph selection.</div>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm">Handling Missing Values</div>

                                            {/* Remove Duplicates Checkbox (Only shows if a file is selected) */}
                                            {trainFile && (
                                                <div className="flex items-center text-xs cursor-pointer">

                                                    <Checkbox
                                                        checked={removeDuplicates}
                                                        onCheckedChange={(checked) => setRemoveDuplicates(!!checked)} // Ensures it's always a boolean
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
                                                            <span>{method}</span>
                                                        </label>
                                                    ))}
                                                </div>
                                            ) : (
                                                <div className="text-center">
                                                    Please select a train file to enable options.
                                                </div>
                                            )}
                                        </div>


                                    </div>






                                </div>
                                <div className="flex gap-x-3">


                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-1 mt-1">Model Configuration</div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">

                                            {trainFile ? (
                                                <>
                                                    <div className="text-[10px] text-gray-400 mb-1">Encoding:</div>
                                                    <Select onValueChange={setEncodingMethod} value={encodingMethod}>
                                                        <SelectTrigger className="w-full text-xs text-white">
                                                            <SelectValue placeholder="Encoding" />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            <SelectItem value="none">None (Numeric Only)</SelectItem>
                                                            <SelectItem value="one-hot">One-Hot</SelectItem>
                                                            <SelectItem value="label">Label</SelectItem>
                                                            <SelectItem value="target">Target</SelectItem>
                                                        </SelectContent>
                                                    </Select>

                                                    <div className="text-[10px] text-gray-400 mb-1 mt-2">Regularization:</div>
                                                    <div className="flex gap-2">
                                                        <Select onValueChange={setRegularizationType} value={regularizationType}>
                                                            <SelectTrigger className="w-2/3 text-xs text-white">
                                                                <SelectValue placeholder="Type" />
                                                            </SelectTrigger>
                                                            <SelectContent>
                                                                <SelectItem value="none">None</SelectItem>
                                                                <SelectItem value="ridge">Ridge</SelectItem>
                                                                <SelectItem value="lasso">Lasso</SelectItem>
                                                                <SelectItem value="elasticnet">ElasticNet</SelectItem>
                                                            </SelectContent>
                                                        </Select>
                                                        <Input 
                                                            type="number" 
                                                            placeholder="α"
                                                            value={alphaValue}
                                                            onChange={(e) => setAlphaValue(e.target.value)}
                                                            className="w-1/3 text-xs"
                                                            step="0.1"
                                                            min="0.001"
                                                            disabled={regularizationType === "none"}
                                                        />
                                                    </div>

                                                    <div className="text-[10px] text-gray-400 mb-1 mt-2">Cross-Validation:</div>
                                                    <div className="flex gap-2 items-center">
                                                        <Checkbox
                                                            checked={enableCV}
                                                            onCheckedChange={(checked) => setEnableCV(!!checked)}
                                                            className="h-3 w-3"
                                                        />
                                                        <span className="text-xs">Enable</span>
                                                        <Input 
                                                            type="number" 
                                                            placeholder="Folds"
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

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm ">Outlier Removal</div>
                                            {trainFile && (
                                                <div className="flex items-center text-xs cursor-pointer">
                                                    <Checkbox
                                                        checked={enableOutlierDetection}
                                                        onCheckedChange={(checked) => setEnableOutlierDetection(!!checked)}
                                                        className="mr-2"
                                                    />
                                                    <span>Enable Outlier Removal</span>
                                                </div>
                                            )}
                                        </div>

                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <>
                                                    {enableOutlierDetection ? (
                                                        <>
                                                            {/* Outlier Detection Method Selection */}
                                                            <Select onValueChange={setOutlierMethod} value={outlierMethod}>
                                                                <SelectTrigger className="w-full mt-3 text-xs text-white">
                                                                    <SelectValue placeholder="Select Outlier Method" />
                                                                </SelectTrigger>
                                                                <SelectContent>
                                                                    <SelectItem value="z-score">Z-Score Method</SelectItem>
                                                                    <SelectItem value="iqr">IQR Method</SelectItem>
                                                                    <SelectItem value="winsorization">Winsorization</SelectItem>
                                                                </SelectContent>
                                                            </Select>

                                                            {/* Method-Specific Inputs */}
                                                            {outlierMethod === "z-score" && (
                                                                <div className="mt-2">
                                                                    <span className="text-xs text-white">Z-Score Threshold:</span>
                                                                    <Input
                                                                        type="number"
                                                                        className="mt-1 w-full text-xs"
                                                                        value={zScoreThreshold}
                                                                        onChange={(e) => setZScoreThreshold(parseFloat(e.target.value))}
                                                                    />
                                                                </div>
                                                            )}

                                                            {outlierMethod === "iqr" && (
                                                                <div className="mt-2">
                                                                    <span className="text-xs text-white">IQR Multiplier (Lower bound & Upper bound):</span>
                                                                    <div className="flex space-x-2">
                                                                        <Input
                                                                            type="number"
                                                                            className="mt-1 w-1/2 text-xs"
                                                                            value={iqrLower}
                                                                            onChange={(e) => setIqrLower(parseFloat(e.target.value))}
                                                                        />
                                                                        <Input
                                                                            type="number"
                                                                            className="mt-1 w-1/2 text-xs"
                                                                            value={iqrUpper}
                                                                            onChange={(e) => setIqrUpper(parseFloat(e.target.value))}
                                                                        />
                                                                    </div>
                                                                </div>
                                                            )}

                                                            {outlierMethod === "winsorization" && (
                                                                <div className="mt-2">
                                                                    <span className="text-xs text-white">Winsorization Percentiles:</span>
                                                                    <div className="flex space-x-2">
                                                                        <Input
                                                                            type="number"
                                                                            className="mt-1 w-1/2 text-xs"
                                                                            value={winsorLower}
                                                                            onChange={(e) => setWinsorLower(parseFloat(e.target.value))}
                                                                        />
                                                                        <Input
                                                                            type="number"
                                                                            className="mt-1 w-1/2 text-xs"
                                                                            value={winsorUpper}
                                                                            onChange={(e) => setWinsorUpper(parseFloat(e.target.value))}
                                                                        />
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </>
                                                    ) : (
                                                        <div className="text-center text-white">
                                                            Enable Outlier Removal to select a method.
                                                        </div>
                                                    )}
                                                </>
                                            ) : (
                                                <div className="text-center text-white">
                                                    Please select a train file to choose for outlier detection.
                                                </div>
                                            )}
                                        </div>
                                    </div>






                                    {/* Advanced Tab */}
                                    <div className="dark:bg-[#212628] h-auto rounded-xl w-1/3 bg-white p-2">
                                        {/* Section Title */}
                                        <div className="font-semibold text-sm mb-2">Feature Scaling</div>

                                        {trainFile ? (
                                            <div className="grid grid-cols-1 gap-3">
                                                {/* Feature Scaling (Single Selection) */}
                                                <div>
                                                    <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] p-3 rounded-lg min-h-[160px] overflow-auto">
                                                        <div className="flex flex-col gap-1">
                                                            {availableFeatureScaling.map((technique) => (
                                                                <div key={technique} className="flex items-center text-xs">
                                                                    <Checkbox
                                                                        checked={selectedFeatureScaling === technique}
                                                                        onCheckedChange={() =>
                                                                            setSelectedFeatureScaling(selectedFeatureScaling === technique ? null : technique)
                                                                        }
                                                                    />
                                                                    <span className="ml-1">{technique}</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </div>
                                                {/* Feature Engineering (Multi Selection) */}
                                                <div>
                                                </div>
                                            </div>
                                        ) : (
                                            // Show this message if no train file is selected
                                            <div className="text-center ">
                                                <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                                    Please select a train file to enable Feature Scaling options.
                                                </div>
                                            </div>
                                        )}
                                    </div>


                                </div>

                                {/* Fourth Row - Feature Comparison Selection */}
                                {hasEffectGraphsSelected && trainFile && selectedTrainColumns.length > 0 && (
                                    <div className="flex gap-x-3">
                                        <div className="dark:bg-[#212628] h-52 rounded-xl w-full bg-white p-2">
                                            <div className="flex items-center justify-between mb-1 mt-1">
                                                <div className="font-semibold text-sm">
                                                    📊 Select Features for Effect Plot Comparisons
                                                    <span className="text-xs font-normal text-gray-500 ml-2">
                                                        (Choose which features to compare against {selectedOutputColumn || 'target'})
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

                        {/* Terminal Tab Content */}
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
                                                                errorDiv.className = 'text-xs text-gray-500 p-2 break-all';
                                                                errorDiv.textContent = `Saved at: ${graphPath}`;
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
                                                ? `Graphs will be saved in: ${datasetPath.replace(/[/\\]+$/, "")}/linearregression-${trainFile.split(".")[0]}`
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
                                            <h2 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                                                📊 Model Performance Metrics
                                            </h2>
                                            <p className="text-sm text-gray-500 dark:text-gray-400">Generated on {new Date().toLocaleString()}</p>
                                        </div>
                                        
                                        {/* Check if results contain the comprehensive table */}
                                        {results.includes("COMPREHENSIVE RESULTS TABLE") || results.includes("MODEL PERFORMANCE SUMMARY") || results.includes("MODEL RESULTS") ? (
                                            <div className="w-full space-y-6">
                                                {/* Parse and display the table beautifully */}
                                                {(() => {
                                                    const lines = results.split("\n");
                                                    const hasCV = results.includes("COMPREHENSIVE RESULTS TABLE");
                                                    
                                                    // For CV: Extract table rows (exclude separator lines)
                                                    const tableLines = lines.filter(line => {
                                                        const trimmed = line.trim();
                                                        // Only include lines with actual data, not separators
                                                        return (trimmed.startsWith("CV Fold") || trimmed.startsWith("Final Model")) && 
                                                               !line.match(/^-+$/);
                                                    });
                                                    
                                                    // Extract CV Statistics
                                                    const cvStatsLines = lines.filter(line => 
                                                        line.includes("Mean R² Score:") || 
                                                        line.includes("Mean MSE:") ||
                                                        line.includes("Model Stability:")
                                                    );
                                                    
                                                    // Extract metrics (works for both CV and non-CV)
                                                    const metricsObj: any = {};
                                                    lines.forEach(line => {
                                                        // Clean the line and split by multiple spaces
                                                        const cleanLine = line.replace(/-+/g, '').trim();
                                                        
                                                        if (cleanLine.includes("R² Score") && !cleanLine.includes("Mean") && !cleanLine.includes("Adjusted")) {
                                                            const parts = cleanLine.split(/\s{2,}|\s*:\s*/);
                                                            if (parts.length >= 2) {
                                                                metricsObj.r2 = parts[1].trim();
                                                            }
                                                        }
                                                        if (cleanLine.includes("Adjusted R² Score")) {
                                                            const parts = cleanLine.split(/\s{2,}|\s*:\s*/);
                                                            if (parts.length >= 2) {
                                                                metricsObj.adj_r2 = parts[parts.length - 1].trim();
                                                            }
                                                        }
                                                        if (cleanLine.includes("Mean Squared Error")) {
                                                            const parts = cleanLine.split(/\s{2,}|\s*:\s*/);
                                                            if (parts.length >= 2) {
                                                                metricsObj.mse = parts[parts.length - 1].trim();
                                                            }
                                                        }
                                                        if (cleanLine.includes("Mean Absolute Error") && !cleanLine.includes("Percentage")) {
                                                            const parts = cleanLine.split(/\s{2,}|\s*:\s*/);
                                                            if (parts.length >= 2) {
                                                                metricsObj.mae = parts[parts.length - 1].trim();
                                                            }
                                                        }
                                                        if (cleanLine.includes("Root Mean Squared Error")) {
                                                            const parts = cleanLine.split(/\s{2,}|\s*:\s*/);
                                                            if (parts.length >= 2) {
                                                                metricsObj.rmse = parts[parts.length - 1].trim();
                                                            }
                                                        }
                                                        if (cleanLine.includes("Mean Absolute Percentage Error")) {
                                                            const parts = cleanLine.split(/\s{2,}|\s*:\s*/);
                                                            if (parts.length >= 2) {
                                                                metricsObj.mape = parts[parts.length - 1].trim();
                                                            }
                                                        }
                                                        if (cleanLine.includes("Accuracy") && !cleanLine.includes("Mean")) {
                                                            const parts = cleanLine.split(/\s{2,}|\s*:\s*/);
                                                            if (parts.length >= 2) {
                                                                metricsObj.accuracy = parts[1].trim();
                                                            }
                                                        }
                                                        if (cleanLine.includes("Training Samples")) {
                                                            const parts = cleanLine.split(/\s{2,}|\s*:\s*/);
                                                            if (parts.length >= 2) {
                                                                metricsObj.trainSize = parts[parts.length - 1].trim();
                                                            }
                                                        }
                                                        if (cleanLine.includes("Test Samples")) {
                                                            const parts = cleanLine.split(/\s{2,}|\s*:\s*/);
                                                            if (parts.length >= 2) {
                                                                metricsObj.testSize = parts[parts.length - 1].trim();
                                                            }
                                                        }
                                                    });
                                                    
                                                    // Parse model rows
                                                    const modelRows = tableLines.filter(line => 
                                                        line.includes("CV Fold") || line.includes("Final Model")
                                                    );
                                                    
                                                    return (
                                                        <>
                                                            {/* Models Table - Only show if CV enabled */}
                                                            {hasCV && modelRows.length > 0 && (
                                                                <div className="dark:bg-[#1a1a1a] bg-white rounded-2xl shadow-2xl overflow-hidden border border-purple-500/30">
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
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">R²</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Adj R²</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">MSE</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">MAE</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">RMSE</th>
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">MAPE</th>
                                                                                {tableLines[0]?.includes("Accuracy") && (
                                                                                    <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Accuracy</th>
                                                                                )}
                                                                                <th className="px-3 py-4 text-center text-xs font-bold text-white uppercase tracking-wider">Train</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                                                                            {modelRows.map((row, idx) => {
                                                                                // Clean the row - remove separator dashes and trim
                                                                                const cleanRow = row.replace(/-+/g, '').trim();
                                                                                const parts = cleanRow.split(/\s+/).filter(p => p.length > 0);
                                                                                const isFinal = row.includes("Final Model");
                                                                                const rowBg = isFinal 
                                                                                    ? "bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20" 
                                                                                    : idx % 2 === 0 
                                                                                        ? "bg-gray-50 dark:bg-gray-800/50" 
                                                                                        : "bg-white dark:bg-gray-900/50";
                                                                                
                                                                                // Determine model name and where metrics start
                                                                                let modelName = '';
                                                                                let metricsStart = 0;
                                                                                
                                                                                if (isFinal) {
                                                                                    // "Final Model (Full)" - first 3 parts
                                                                                    modelName = parts.slice(0, 3).join(' ');
                                                                                    metricsStart = 3;
                                                                                } else {
                                                                                    // "CV Fold 1" - first 3 parts
                                                                                    modelName = parts.slice(0, 3).join(' ');
                                                                                    metricsStart = 3;
                                                                                }
                                                                                
                                                                                return (
                                                                                    <tr key={idx} className={`${rowBg} hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-colors`}>
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
                                                                                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
                                                                                                {parts[metricsStart + 1]?.trim() || 'N/A'}
                                                                                            </span>
                                                                                        </td>
                                                                                        <td className="px-3 py-4 text-center">
                                                                                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200">
                                                                                                {parts[metricsStart + 2]?.trim() || 'N/A'}
                                                                                            </span>
                                                                                        </td>
                                                                                        <td className="px-3 py-4 text-center">
                                                                                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                                                                                                {parts[metricsStart + 3]?.trim() || 'N/A'}
                                                                                            </span>
                                                                                        </td>
                                                                                        <td className="px-3 py-4 text-center">
                                                                                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                                                                                                {parts[metricsStart + 4]?.trim() || 'N/A'}
                                                                                            </span>
                                                                                        </td>
                                                                                        <td className="px-3 py-4 text-center">
                                                                                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200">
                                                                                                {parts[metricsStart + 5]?.trim() || 'N/A'}
                                                                                            </span>
                                                                                        </td>
                                                                                        {tableLines[0]?.includes("Accuracy") && (
                                                                                            <td className="px-3 py-4 text-center">
                                                                                                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                                                                                                    {parts[metricsStart + 6]?.trim() || 'N/A'}
                                                                                                </span>
                                                                                            </td>
                                                                                        )}
                                                                                        <td className="px-3 py-4 text-center">
                                                                                            <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">
                                                                                                {parts[tableLines[0]?.includes("Accuracy") ? metricsStart + 7 : metricsStart + 6]?.trim() || 'N/A'}
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
                                                            
                                                            {/* CV Statistics Cards */}
                                                            {cvStatsLines.length > 0 && (
                                                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                                                    {cvStatsLines.map((line, idx) => {
                                                                        const [label, value] = line.split(":").map(s => s.trim());
                                                                        const icon = label?.includes("R²") ? "📊" : label?.includes("MSE") ? "⚠️" : "✓";
                                                                        const color = label?.includes("R²") ? "from-purple-500 to-purple-600" 
                                                                            : label?.includes("MSE") ? "from-orange-500 to-orange-600"
                                                                            : "from-green-500 to-green-600";
                                                                        
                                                                        return (
                                                                            <div key={idx} className={`bg-gradient-to-br ${color} p-6 rounded-2xl shadow-xl text-white`}>
                                                                                <div className="text-4xl mb-2">{icon}</div>
                                                                                <div className="text-sm opacity-90 mb-1">{label}</div>
                                                                                <div className="text-2xl font-bold">{value}</div>
                                                                            </div>
                                                                        );
                                                                    })}
                                                                </div>
                                                            )}
                                                            
                                                            {/* Final Model Performance - Always show */}
                                                            {(metricsObj.r2 || metricsObj.mse) && (
                                                                <div className="dark:bg-[#1a1a1a] bg-white rounded-2xl shadow-2xl p-8 border-2 border-green-500/50">
                                                                    <h3 className="text-2xl font-bold mb-6 flex items-center gap-3 text-green-600 dark:text-green-400">
                                                                        <span className="text-3xl">🎯</span>
                                                                        {hasCV ? 'Final Model Performance' : 'Model Performance'}
                                                                    </h3>
                                                                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                                                                        {metricsObj.r2 && (
                                                                            <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-6 rounded-xl border border-purple-300 dark:border-purple-700">
                                                                                <div className="text-xs text-purple-600 dark:text-purple-400 mb-1 font-semibold">R² Score</div>
                                                                                <div className="text-3xl font-bold text-purple-700 dark:text-purple-300">{metricsObj.r2}</div>
                                                                            </div>
                                                                        )}
                                                                        {metricsObj.adj_r2 && (
                                                                            <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-900/20 dark:to-indigo-800/20 p-6 rounded-xl border border-indigo-300 dark:border-indigo-700">
                                                                                <div className="text-xs text-indigo-600 dark:text-indigo-400 mb-1 font-semibold">Adjusted R²</div>
                                                                                <div className="text-3xl font-bold text-indigo-700 dark:text-indigo-300">{metricsObj.adj_r2}</div>
                                                                            </div>
                                                                        )}
                                                                        {metricsObj.mse && (
                                                                            <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 p-6 rounded-xl border border-orange-300 dark:border-orange-700">
                                                                                <div className="text-xs text-orange-600 dark:text-orange-400 mb-1 font-semibold">MSE</div>
                                                                                <div className="text-3xl font-bold text-orange-700 dark:text-orange-300">{metricsObj.mse}</div>
                                                                            </div>
                                                                        )}
                                                                        {metricsObj.mae && (
                                                                            <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 dark:from-yellow-900/20 dark:to-yellow-800/20 p-6 rounded-xl border border-yellow-300 dark:border-yellow-700">
                                                                                <div className="text-xs text-yellow-600 dark:text-yellow-400 mb-1 font-semibold">MAE</div>
                                                                                <div className="text-3xl font-bold text-yellow-700 dark:text-yellow-300">{metricsObj.mae}</div>
                                                                            </div>
                                                                        )}
                                                                        {metricsObj.rmse && (
                                                                            <div className="bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 p-6 rounded-xl border border-red-300 dark:border-red-700">
                                                                                <div className="text-xs text-red-600 dark:text-red-400 mb-1 font-semibold">RMSE</div>
                                                                                <div className="text-3xl font-bold text-red-700 dark:text-red-300">{metricsObj.rmse}</div>
                                                                            </div>
                                                                        )}
                                                                        {metricsObj.mape && (
                                                                            <div className="bg-gradient-to-br from-pink-50 to-pink-100 dark:from-pink-900/20 dark:to-pink-800/20 p-6 rounded-xl border border-pink-300 dark:border-pink-700">
                                                                                <div className="text-xs text-pink-600 dark:text-pink-400 mb-1 font-semibold">MAPE</div>
                                                                                <div className="text-3xl font-bold text-pink-700 dark:text-pink-300">{metricsObj.mape}</div>
                                                                            </div>
                                                                        )}
                                                                        {metricsObj.accuracy && (
                                                                            <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-6 rounded-xl border border-green-300 dark:border-green-700">
                                                                                <div className="text-xs text-green-600 dark:text-green-400 mb-1 font-semibold">Accuracy</div>
                                                                                <div className="text-3xl font-bold text-green-700 dark:text-green-300">{metricsObj.accuracy}</div>
                                                                            </div>
                                                                        )}
                                                                        {metricsObj.trainSize && (
                                                                            <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-xl border border-gray-300 dark:border-gray-700">
                                                                                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1 font-semibold">Training Samples</div>
                                                                                <div className="text-2xl font-bold text-gray-900 dark:text-white">{metricsObj.trainSize}</div>
                                                                            </div>
                                                                        )}
                                                                        {metricsObj.testSize && (
                                                                            <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-xl border border-gray-300 dark:border-gray-700">
                                                                                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1 font-semibold">Test Samples</div>
                                                                                <div className="text-2xl font-bold text-gray-900 dark:text-white">{metricsObj.testSize}</div>
                                                                            </div>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </>
                                                    );
                                                })()}
                                            </div>
                                        ) : (
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
                                                {results
                                                    .split("\n")
                                                    .filter((line) => line.trim() !== "")
                                                    .map((line, index) => {
                                                    const parts = line.split(":");
                                                    const label = parts[0]?.trim();
                                                    const value = parts[1]?.trim();
                                                    
                                                    // Determine icon and color based on metric
                                                    let icon = "📈";
                                                    let bgColor = "from-blue-500 to-blue-600";
                                                    
                                                    if (label?.includes("Accuracy")) {
                                                        icon = "🎯";
                                                        bgColor = "from-green-500 to-green-600";
                                                    } else if (label?.includes("R-squared")) {
                                                        icon = "📊";
                                                        bgColor = "from-purple-500 to-purple-600";
                                                    } else if (label?.includes("Error") || label?.includes("MSE")) {
                                                        icon = "⚠️";
                                                        bgColor = "from-orange-500 to-orange-600";
                                                    }
                                                    
                                                    return (
                                                        <div 
                                                            key={index} 
                                                            className={`bg-gradient-to-br ${bgColor} p-6 rounded-2xl shadow-xl text-white transform hover:scale-105 transition-transform duration-200`}
                                                        >
                                                            <div className="flex items-center justify-between mb-3">
                                                                <span className="text-4xl">{icon}</span>
                                                                <div className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-xs font-semibold">
                                                                    Metric {index + 1}
                                                                </div>
                                                            </div>
                                                            <div className="text-lg font-medium mb-2 opacity-90">
                                                                {label}
                                                            </div>
                                                            <div className="text-4xl font-bold">
                                                                {value || "N/A"}
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        )}
                                        
                                        {results.toLowerCase().includes("skipping") && (
                                            <div className="mt-6 p-4 bg-yellow-100 dark:bg-yellow-900 border-l-4 border-yellow-500 rounded-lg">
                                                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                                                    ℹ️ Note: Some metrics are not applicable for this type of analysis.
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="text-center space-y-4">
                                        <div className="text-6xl mb-4">📊</div>
                                        <h3 className="text-3xl font-bold bg-gradient-to-r from-gray-600 to-gray-800 dark:from-gray-300 dark:to-gray-500 bg-clip-text text-transparent">
                                            Waiting for Results
                                        </h3>
                                        <p className="text-gray-500 dark:text-gray-400">
                                            Run the model to see performance metrics here
                                        </p>
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        {/* Predict Tab Content */}
                        <TabsContent value="predict">
                            <div className="border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-6">
                                {modelTrained ? (
                                    <div className="space-y-6">
                                        <div className="text-center mb-6">
                                            <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                                                🔮 Make Predictions
                                            </h2>
                                            <p className="text-sm text-gray-500 dark:text-gray-400">
                                                Enter feature values to predict {selectedOutputColumn || 'the target'}
                                            </p>
                                        </div>

                                        {/* Model Selector */}
                                        {availableModels.length >= 1 && (
                                            <div className="dark:bg-[#212628] bg-white p-4 rounded-lg border border-purple-500/30">
                                                <Label className="text-sm font-semibold mb-2 block">Select Model for Prediction:</Label>
                                                <Select onValueChange={setSelectedModel} value={selectedModel}>
                                                    <SelectTrigger className="w-full">
                                                        <SelectValue placeholder="Choose a model" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        {availableModels.map((model) => (
                                                            <SelectItem key={model.filename} value={model.filename}>
                                                                {model.name} - R²: {model.r2_score.toFixed(4)} | MSE: {model.mse.toFixed(4)}
                                                                {model.accuracy !== null && model.accuracy !== undefined && ` | Acc: ${model.accuracy.toFixed(4)}`}
                                                            </SelectItem>
                                                        ))}
                                                    </SelectContent>
                                                </Select>
                                                <p className="text-xs text-gray-500 mt-2">
                                                    {availableModels.length} {availableModels.length === 1 ? 'model' : 'models'} available
                                                    {availableModels.find(m => m.filename === selectedModel)?.type === 'cv_fold' && 
                                                        ` | Using CV Fold ${availableModels.find(m => m.filename === selectedModel)?.fold_number}`
                                                    }
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
                                                            {isCategorical && <span className="ml-2 text-xs text-purple-500 font-medium">(categorical)</span>}
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
                                                                className="w-full px-3 py-2 border rounded-lg dark:bg-[#0F0F0F] dark:border-gray-700 dark:text-white focus:ring-2 focus:ring-purple-500"
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
                                                        alert(`Please fill in all fields. Missing: ${missingFields.join(", ")}`);
                                                        return;
                                                    }

                                                    setIsPredicting(true);
                                                    setPredictionResult(null);

                                                    try {
                                                        // Construct the path to the selected model
                                                        const normalizedPath = datasetPath.trim().replace(/[/\\]+$/, "");
                                                        const isWindows = navigator.platform.startsWith("Win");
                                                        const separator = isWindows ? "\\\\" : "/";
                                                        const modelDir = `${normalizedPath}${separator}linearregression-${trainFile?.split(".")[0]}`;
                                                        const modelPath = `${modelDir}${separator}${selectedModel}`;

                                                        // Prepare input values as an object with feature names
                                                        // Keep text values as strings, convert numbers
                                                        const inputValues: Record<string, any> = {};
                                                        selectedTrainColumns.forEach(col => {
                                                            const val = predictionInputs[col];
                                                            // Keep original value (text or number)
                                                            inputValues[col] = val;
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
                                                        
                                                        console.log("Prediction response:", data);
                                                        
                                                        if (data.error) {
                                                            setPredictionResult(`Error: ${data.error}`);
                                                            setPredictionIsBinary(null);
                                                        } else {
                                                            setPredictionResult(data.prediction);
                                                            setPredictionIsBinary(data.is_binary ?? null);
                                                        }
                                                    } catch (error) {
                                                        setPredictionResult(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
                                                    } finally {
                                                        setIsPredicting(false);
                                                    }
                                                }}
                                                disabled={isPredicting}
                                                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-8 py-3 rounded-lg font-semibold"
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
                                                <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-8 rounded-2xl shadow-2xl text-white text-center transform hover:scale-105 transition-transform duration-200">
                                                    <div className="text-6xl mb-4">🎯</div>
                                                    <div className="text-xl font-medium mb-3 opacity-90">
                                                        Predicted {selectedOutputColumn}
                                                    </div>
                                                    <div className="text-6xl font-bold mb-4">
                                                        {predictionResult}
                                                    </div>
                                                    
                                                    {/* Smart interpretation - Binary classification vs Regression */}
                                                    {(() => {
                                                        const value = parseFloat(predictionResult);
                                                        if (!isNaN(value)) {
                                                            // Use backend's determination of binary classification
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
                                                            // Regression (continuous value) or unknown
                                                            else if (predictionIsBinary === false) {
                                                                return (
                                                                    <div className="mt-4 space-y-2">
                                                                        <div className="text-sm opacity-75">
                                                                            📈 Regression Prediction
                                                                        </div>
                                                                        <div className="text-xs opacity-60">
                                                                            Continuous value predicted by the model
                                                                        </div>
                                                                    </div>
                                                                );
                                                            } else {
                                                                // Fallback when binary info not available
                                                                return (
                                                                    <div className="text-sm opacity-75 mt-2">
                                                                        Predicted value
                                                                    </div>
                                                                );
                                                            }
                                                        }
                                                        return null;
                                                    })()}
                                                    
                                                    <div className="text-sm opacity-75 mt-4">
                                                        Based on the trained model
                                                    </div>
                                                </div>
                                            </div>
                                        )}

                                        <div className="mt-6 space-y-3">
                                            <div className="p-4 bg-blue-100 dark:bg-blue-900 border-l-4 border-blue-500 rounded-lg">
                                                <p className="text-sm text-blue-800 dark:text-blue-200">
                                                    💡 <strong>Tip:</strong> <span className="text-purple-600 dark:text-purple-300 font-semibold">Categorical features</span> appear as dropdowns with values from training data. <span className="text-blue-600 dark:text-blue-300 font-semibold">Numeric features</span> require manual number entry.
                                                </p>
                                            </div>
                                            
                                            <div className="p-4 bg-purple-100 dark:bg-purple-900 border-l-4 border-purple-500 rounded-lg">
                                                <p className="text-sm text-purple-800 dark:text-purple-200">
                                                    📊 <strong>Result Types:</strong>
                                                </p>
                                                <ul className="text-xs text-purple-700 dark:text-purple-300 mt-2 ml-4 space-y-1">
                                                    <li>• <strong>Binary (0-1):</strong> Classification result with percentage confidence</li>
                                                    <li>• <strong>Continuous (any value):</strong> Regression result for numeric predictions like price, age, etc.</li>
                                                    <li>• The model automatically detects which type based on the output range</li>
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
                                        <Button
                                            onClick={handleRunScript}
                                            disabled={!trainFile || selectedTrainColumns.length === 0 || !selectedOutputColumn}
                                            className="mt-6 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-8 py-3 rounded-lg font-semibold"
                                        >
                                            ▶️ Train Model
                                        </Button>
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

export default LinearRegressionComponent;
