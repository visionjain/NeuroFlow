"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import axios from "axios";
import { useParams } from "next/navigation";
import Nav from "@/components/navbar/page";
import CopyRight from "@/components/copybar/page";
import Loader from "@/components/loader/page";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { FaWandMagicSparkles } from "react-icons/fa6";
import { FaMicrophoneAlt } from "react-icons/fa";
import { toast } from 'sonner';
import ReactMarkdown from "react-markdown";
import { saveAs } from "file-saver";
import { Document, Packer, Paragraph, TextRun } from "docx";


import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"

declare global {
  interface Window {
    webkitSpeechRecognition: any;
  }
}

const ProjectPage = () => {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [userDetails, setUserDetails] = useState<any>(null);
  const [userRole, setUserRole] = useState("");
  const { id } = useParams(); // Get the project ID from the URL
  const [projectDetails, setProjectDetails] = useState<any>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [time, setTime] = useState(0); // Time in seconds
  const [transcript, setTranscript] = useState(""); // Store the final live transcript
  const [recognition, setRecognition] = useState<any>(null);
  const [notes, setNotes] = useState(null);
  const [qwiz, setQwiz] = useState(null);
  const [flashcards, setFlashcards] = useState(null);
  const [cheatSheet, setCheatSheet] = useState(null);
  const [buttonText, setButtonText] = useState("Generate");
  const [isGenerating, setIsGenerating] = useState(false);
  const [buttonAnimation, setButtonAnimation] = useState("");


  // Speech Recognition Setup
  

  const formatProjectTime = (isoString: string) => {
    const date = new Date(isoString);
    const formattedDate = date.toLocaleDateString("en-IN"); // Format as DD/MM/YYYY
    const formattedTime = date.toLocaleTimeString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false, // 24-hour format
    });
    return `${formattedDate} ${formattedTime}`;
  };

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const response = await axios.get("/api/users/me");
        const { data } = response.data;
        setUserDetails(data);
        setUserRole(data.role);
        setLoading(false);

        // Find the project based on the ID
        const project = data.projects.find(
          (project: any) => project._id.toString() === id
        );

        if (project) {
          setProjectDetails({
            projectId: project._id,
            projectName: project.topic,
            ProjectTime: formatProjectTime(project.createdAt),
          });

          // Set the transcript to the state if it exists
          setTranscript(project.transcript || "");
          setNotes(project.notes || "");
          setQwiz(project.qwiz || "");
          setFlashcards(project.flashcards || "");
          setCheatSheet(project.cheatSheet || "");
        } else {
          console.error("Project not found");
        }
      } catch (error) {
        expirylogout();
        router.push("/login");
      }
    };

    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
    } else {
      fetchUserData();
    }
  }, [router, id]);


  const expirylogout = async () => {
    try {
      await axios.get("/api/users/logout");
      localStorage.removeItem("token");
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  if (!projectDetails) {
    return (
      <div className="flex h-screen justify-center items-center">
        <Loader />
      </div>
    );
  }

  return (
    <div>
      <div className="h-[100vh] pt-24">
        <Nav loading={loading} userRole={userRole} userDetails={userDetails} />
        <div
          className="h-[90%] dark:bg-[#212628] rounded-3xl ml-8 bg-white mr-8 overflow-y-auto"
          style={{ maxHeight: "90vh" }}
        >
          <div>
            <div className="text-xl ">
              <div className="pl-4 pt-4 font-bold">
                <h1 className="italic text-3xl">
                  {projectDetails.projectName} {" "}
                  <span className="text-base lowercase">
                    {projectDetails.ProjectTime}
                  </span>
                </h1>
              </div>

              <div
                className={`h-16 border border-[rgb(61,68,77)] flex justify-center mt-4 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4`}
              >
              </div>

            </div>
          </div>
        </div>
        <CopyRight />
      </div>
    </div>
  );
};

export default ProjectPage;
