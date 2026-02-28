README file.
This is the explanation of the program to present.

🧠 Overview
EduPulse is a real-time classroom analytics and AI-assisted learning platform that enables teachers to:
Monitor student understanding live


Detect confusion patterns


Provide adaptive hints automatically


Identify potential plagiarism or copy-paste behavior


Generate post-class performance reports


The system transforms traditional reactive teaching into data-driven adaptive teaching.

🎯 Problem Statement
In traditional classrooms:
Teachers cannot see who truly understands the topic.


Struggling students remain silent.


Some students copy solutions without learning.


There is no measurable understanding metric.


Post-class evaluation is delayed and incomplete.


EduPulse solves this by introducing real-time behavioral and performance analytics.

🏗 System Architecture (High-Level)
Teacher Dashboard
       ↓
Backend API + WebSocket Server
       ↓
Real-Time Event Engine
       ↓
AI Assistance Engine
       ↓
Student Interface
Core components:
Teacher Interface


Student Workspace


Real-Time Monitoring Engine


AI Hint & Correction Module


Academic Integrity Detector


Post-Class Analytics Module



👨‍🏫 Teacher Workflow
Step 1 – Create Class Session
Teacher:
Logs in


Clicks “Start New Session”


Uploads:


Lecture slides (PDF)


Coding tasks / Exercises


Expected solutions (optional)


System generates:
Session ID


Student access link / QR code



Step 2 – Students Join
Students:
Join session via link


Get:


Task list


Interactive workspace (text editor / coding environment)


System starts tracking live metrics.

👨‍🎓 Student Workspace (Core of the System)
Each student has:
Task panel


Code editor / answer input


“I’m confused” button


Hint request button


While working, the system tracks:
Typing behavior


Edit frequency


Idle time


Copy-paste events


Error rate


Hint requests


Submission attempts


This creates a real-time behavioral profile.

🔴 Real-Time Monitoring Engine
The backend continuously calculates:
1️⃣ Understanding Score
Based on:
Task completion speed


Error rate


Hint frequency


Correction iterations


Example formula:
Understanding Score =
(Completion Progress * 0.4)
+ (Accuracy * 0.3)
- (Hint Usage * 0.2)
- (Idle Time * 0.1)
Displayed as:
🟢 Good understanding


🟡 Struggling


🔴 Critical confusion



2️⃣ Confusion Spike Detection
If multiple students:
Pause for long time


Make same mistake


Request hints at same moment


System detects:
“Topic Confusion Cluster Detected: Recursion Base Case”
Teacher receives live alert.

🤖 AI Assistance Engine
When a student struggles:
Trigger conditions:
3 failed attempts


Long inactivity


Repeated same error


System automatically:
Analyzes student attempt


Compares with expected logic


Provides progressive hint:


Level 1 → Concept hint
 Level 2 → Structural hint
 Level 3 → Partial solution
This prevents full answer giving.

🚨 Academic Integrity Detection
The system monitors:
Large paste events


Sudden perfect solution insertion


Similarity between student submissions


Detection methods:
Copy-paste length threshold


Code similarity comparison


AI-generated style detection (optional advanced)


If detected:
Flag appears in teacher dashboard


Student marked for review


No automatic punishment



📊 Teacher Dashboard (Live)
Teacher sees:
Real-Time Overview
Student
Progress
Understanding
Struggle Risk
Anna
70%
🟢
Low
Mark
30%
🔴
High


Class-Level Metrics
Average understanding score


Confusion heatmap per topic


Hint usage distribution


Plagiarism alerts


Engagement timeline graph



📈 Post-Class Analytics Report
After session ends, system generates:
Report Includes:
1️⃣ Class Understanding Index
Overall comprehension percentage.
2️⃣ Topic Difficulty Ranking
Which topics caused most confusion.
3️⃣ Individual Student Profiles
Strength areas


Weak areas


Engagement level


Risk indicators


4️⃣ Behavioral Insights
Average idle time


Hint dependency rate


Copy-paste incidents


Delivered as:
PDF report


Downloadable analytics dashboard


Email summary



🧩 Core Functional Modules
1️⃣ Session Manager
Handles class creation and lifecycle.
2️⃣ Real-Time Event Tracker
Captures:
Keystrokes (not content, but behavior)


Submission attempts


Hint requests


Copy events


3️⃣ AI Hint Engine
Analyzes attempt → generates structured feedback.
4️⃣ Similarity Engine
Compares:
Student vs expected solution


Student vs student


5️⃣ Analytics Engine
Processes raw events into:
Metrics


Risk scoring


Reports



🔐 Privacy & Ethics Layer
Important for presentation:
Only behavioral metadata tracked


No intrusive screen recording


Teachers cannot see private browsing


Student consent required


This avoids ethical criticism.

🚀 Advanced Future Extensions
LMS integration (Moodle, Canvas)


Long-term learning analytics


Burnout prediction


AI-powered personalized curriculum


Emotion detection (optional advanced research direction)



💡 Value Proposition
EduPulse shifts education from:
Reactive teaching → Predictive & adaptive learning
Instead of asking:
“Does everyone understand?”
The system shows measurable understanding in real time.

🏆 Why This Wins Hackathons
Real problem


Scalable SaaS potential


Strong AI integration


Real-time architecture


Ethical awareness


Data-driven education


Visual dashboard demo



🎤 30-Second Pitch Version
“EduPulse is a real-time AI-powered classroom intelligence system that tracks student understanding, detects confusion patterns, provides adaptive hints, flags copy-paste behavior, and generates post-class analytics reports — transforming traditional classrooms into data-driven adaptive learning environments.”

