"use client";

import React, { useEffect, useState } from "react";
import CopyRight from "../copybar/page";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle, DialogFooter } from "../ui/dialog";
import { Input } from "../ui/input";
import { useRouter } from "next/navigation";
import { toast } from "sonner"; // Import toast library
import { FaPlus, FaRegEdit, FaTrashAlt } from "react-icons/fa";
import { TfiWrite } from "react-icons/tfi";
import axios from "axios";
import {
    Select,
    SelectContent,
    SelectGroup,
    SelectItem,
    SelectLabel,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"


type Project = {
    _id: string;
    topic: string;
    createdAt?: string;
    algorithm: string;
};

const Home = () => {
    const router = useRouter();
    const [isMounted, setIsMounted] = useState(false);
    const [token, setToken] = useState<string | null>(null);
    const [projects, setProjects] = useState<Project[]>([]);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [newProject, setNewProject] = useState("");
    const [editingProject, setEditingProject] = useState<Project | null>(null);
    const [projectToDelete, setProjectToDelete] = useState<Project | null>(null);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [selectedAlgorithm, setSelectedAlgorithm] = useState("");


    useEffect(() => {
        setIsMounted(true);
        const jwtToken = localStorage.getItem("token");
        setToken(jwtToken);

        // Fetch projects when the component mounts
        if (jwtToken) {
            fetchProjects();
        }
    }, []);

    const fetchProjects = async () => {
        try {
            const response = await axios.get("/api/users/projects", {
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("token")}`,
                },
            });
            setProjects(response.data.projects || []);
        } catch (error) {
            console.error("Error fetching projects:", error);
            toast.error("Failed to fetch projects, please login again", {
                style: {
                    background: 'red',
                    color: 'white',
                },
            });
        }
    };

    const handleLoginRedirect = () => {
        router.push("/login");
    };

    const handleAddProject = async () => {
        if (newProject.trim() === "" || selectedAlgorithm.trim() === "") {
            toast.error("Project name and algorithm cannot be empty", {
                style: { background: "red", color: "white" },
            });
            return;
        }

        try {
            const data = {
                topic: newProject.trim(),
                algorithm: selectedAlgorithm, // Include selected algorithm
            };

            const response = await axios.post("/api/users/project", data, {
                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
            });

            setNewProject("");
            setSelectedAlgorithm(""); // Reset algorithm after submission
            setDialogOpen(false);
            toast.success(response.data.message || "Project added successfully!", {
                style: { background: "green", color: "white" },
            });

            fetchProjects(); // Refresh project list
        } catch (error: any) {
            toast.error(error.response?.data?.error || "Failed to add project", {
                style: { background: "red", color: "white" },
            });
        }
    };


    const handleOpenProject = (projectId: string) => {
        router.push(`/project/${projectId}`);  // Just pass the path
    };


    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter") {
            if (editingProject) {
                handleUpdateProject(editingProject._id);
            } else {
                handleAddProject();
            }
        }
    };

    const handleEditProject = (project: Project) => {
        setEditingProject(project); // Set project to edit
        setNewProject(project.topic); // Set input to current topic
        setDialogOpen(true); // Open dialog
    };

    const handleDeleteProject = async (projectId: string) => {
        try {
            const response = await axios.delete("/api/users/deleteproject", {
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("token")}`,
                },
                data: {
                    projectId: projectId,
                },
            });

            // Handle successful deletion
            setProjects(projects.filter((project) => project._id !== projectId)); // Update state
            toast.success("Project deleted successfully", {
                style: {
                    background: 'green',
                    color: 'white',
                },
            });

            // Close the delete confirmation dialog
            setDeleteDialogOpen(false);

        } catch (error) {
            toast.error("Failed to delete project", {
                style: {
                    background: 'red',
                    color: 'white',
                },
            });
        }
    };



    const handleDeleteConfirmation = (project: Project) => {
        setProjectToDelete(project); // Store the project to be deleted
        setDeleteDialogOpen(true); // Open delete confirmation dialog
    };

    const handleUpdateProject = async (projectId: string) => {
        if (newProject.trim() === "") {
            toast.error("Project topic cannot be empty", {
                style: {
                    background: 'red',
                    color: 'white',
                },
            });
            return;
        }

        try {
            const data = {
                projectId,    // Send the projectId in the request body
                newTopic: newProject.trim(),  // Send the new topic in the request body
            };

            const response = await axios.put(`/api/users/updateproject`, data, {
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("token")}`,
                },
            });

            setNewProject(""); // Clear input
            setDialogOpen(false); // Close dialog
            setEditingProject(null); // Clear editing state
            toast.success(response.data.message || "Project updated successfully!", {
                style: {
                    background: 'green',
                    color: 'white',
                },
            });

            // Fetch updated project
            fetchProjects();
        } catch (error: any) {
            toast.error(error.response?.data?.error || "Failed to update project", {
                style: {
                    background: 'red',
                    color: 'white',
                },
            });
        }
    };

    if (!isMounted) {
        return null;
    }

    return (
        <div className="h-[100vh] pt-24">
            <div className="h-[90%] dark:bg-[#212628] rounded-3xl ml-8 bg-white mr-8">
                {/* Header Section */}
                <div className="flex space-x-4 pl-4 pt-4">
                    <div className="w-[99%] h-16 bg-[#E6E6E6] dark:bg-[#0F0F0F] rounded-3xl flex flex-col justify-center items-center text-center space-y-4">
                        <div>
                            {token ? (
                                <Button
                                    className="rounded-3xl px-20"
                                    onClick={() => {
                                        setEditingProject(null); // Ensure it's a new project
                                        setNewProject(""); // Reset project name
                                        setSelectedAlgorithm(""); // Reset algorithm selection
                                        setDialogOpen(true); // Open the dialog
                                    }}
                                >
                                    <FaPlus className="mr-2" /> New Project
                                </Button>

                            ) : (
                                <Button className="rounded-3xl " onClick={handleLoginRedirect}>
                                    <FaPlus className="mr-2" /> Login To Create
                                </Button>
                            )}
                        </div>
                    </div>
                </div>

                {/* Table Section */}
                <div className="flex space-x-4 pl-4 pt-4">
                    <div className="w-[99%] h-[460px] bg-[#E6E6E6] dark:bg-[#0F0F0F] rounded-3xl p-4 overflow-y-auto">
                        <table className="table-auto w-full text-left border-collapse">
                            <thead>
                                <tr>
                                    <th className="p-2 border-b border-gray-700 w-8"></th>
                                    <th className="p-2 border-b border-gray-700">Projects</th>
                                    <th className="p-2 border-b border-gray-700 text-right"></th>
                                </tr>
                            </thead>
                            <tbody>
                                {projects.map((project) => (
                                    <tr key={project._id} className="hover:bg-gray-200 dark:hover:bg-gray-900">
                                        <td className="p-2 border-b border-gray-700">
                                            <TfiWrite className="text-xl" />
                                        </td>
                                        <td className="p-2 border-b border-gray-700">{project.topic} - {project.algorithm}</td>
                                        <td className="p-2 border-b border-gray-700 text-right flex justify-end space-x-2">
                                            <Button
                                                onClick={() => handleOpenProject(project._id)}
                                                className="flex items-center px-6 mr-2 py-1 text-sm font-medium rounded-lg"
                                            >
                                                Open
                                            </Button>
                                            <button
                                                className="text-xl text-blue-500"
                                                onClick={() => handleEditProject(project)}
                                            >
                                                <FaRegEdit className="mr-1" />
                                            </button>
                                            <button
                                                className="text-xl text-red-500"
                                                onClick={() => handleDeleteConfirmation(project)}
                                            >
                                                <FaTrashAlt className="mr-1" />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                                {projects.length === 0 && (
                                    <tr>
                                        <td colSpan={3} className="p-4 text-center text-gray-500 dark:text-gray-400">
                                            No projects added yet.
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* Delete Confirmation Dialog */}
            <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
                <DialogContent>
                    <DialogTitle>Are you sure you want to delete this project?</DialogTitle>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
                            Cancel
                        </Button>
                        <Button
                            className="ml-2"
                            onClick={() => projectToDelete && handleDeleteProject(projectToDelete._id)}
                        >
                            Confirm
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Dialog for Adding or Editing project */}
            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                <DialogContent>
                    <DialogTitle>{editingProject ? "Edit Project" : "Add New Project"}</DialogTitle>
                    <div className="space-y-4">
                        {/* Project Name Input */}
                        <Input
                            placeholder="Enter project name"
                            value={newProject}
                            onChange={(e) => setNewProject(e.target.value)}
                            onKeyDown={handleKeyDown}
                        />

                        {/* Show Algorithm Selection only when adding a new project */}
                        {!editingProject && (
                            <Select onValueChange={setSelectedAlgorithm} value={selectedAlgorithm}>
                                <SelectTrigger>
                                    <SelectValue placeholder="Select an Algorithm" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectGroup>
                                        <SelectLabel className="dark:text-[#8C9AAE]">Algorithm</SelectLabel>
                                        <SelectItem className="dark:text-[#8C9AAE]" value="Linear regression">Linear Regression</SelectItem>
                                        <SelectItem className="dark:text-[#8C9AAE]" value="Logistic regression">Logistic Regression</SelectItem>
                                        <SelectItem className="dark:text-[#8C9AAE]" value="knn">K-Nearest Neighbor</SelectItem>
                                    </SelectGroup>
                                </SelectContent>
                            </Select>
                        )}
                    </div>

                    {/* Dialog Footer with Buttons */}
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setDialogOpen(false)}>
                            Cancel
                        </Button>
                        <Button
                            onClick={async () => {
                                if (editingProject) {
                                    await handleUpdateProject(editingProject._id);
                                } else {
                                    handleAddProject();
                                }
                            }}
                        >
                            {editingProject ? "Update Project" : "Add Project"}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>


            <CopyRight />
        </div>
    );
};

export default Home;