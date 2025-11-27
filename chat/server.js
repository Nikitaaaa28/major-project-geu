import * as dotenv from 'dotenv';
dotenv.config();

import http from 'http';
import url from 'url';

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({});
// History stores the conversation. 
// NOTE: If testing multiple languages, restart the server to clear this memory.
const History = [] 

// Function to rephrase user questions (Forces English for Search)
async function transformQuery(question){
    // Create a temporary history so we don't mess up the main conversation
    const tempHistory = [...History, { role: 'user', parts: [{ text: question }] }];

    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: tempHistory,
        config: {
          systemInstruction: `You are a query rewriting expert. Rephrase the last user question into a standalone question **IN ENGLISH**.
          Even if the user asks in Hindi/Tamil/Hinglish, translate the intent to English for database search.
          Only output the rewritten English question.`,
        },
    });
     
    return response.text
}

// Main Chat Function
async function chatting(question) {
    const queries = await transformQuery(question);
    
    // 1. Vector Search
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });
    const queryVector = await embeddings.embedQuery(queries); 
 
    // 2. Pinecone Search
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    const searchResults = await pineconeIndex.query({
        topK: 10,
        vector: queryVector,
        includeMetadata: true,
    });

    const context = searchResults.matches
                       .map(match => match.metadata.text)
                       .join("\n\n---\n\n");

    // 3. Generate Response
    History.push({
        role:'user',
        parts:[{text:question}]
    });
    
    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: History,
        config: {
            // --- YOUR EXACT PROMPT (NO CHANGES) ---
            systemInstruction: `You are 'SmartHealthChat', a warm, empathetic, and thorough virtual health advisor. 
Your goal is to simulate a consultation with a caring, human doctor.

**LANGUAGE & SCRIPT RULE (CRITICAL):** **You must answer in the SAME LANGUAGE and SAME SCRIPT the user is using.**
-  **ENGLISH INPUT:** If the user writes in **Standard English** (e.g., "I feel sick", "Vomiting", "Fever"), you **MUST** reply in **STANDARD ENGLISH**. Do NOT switch to Hindi or Hinglish.
- If the user asks in **Hindi (Devanagari script)** like "मुझे बुखार है", you MUST reply in **Hindi (Devanagari script)**.
- If the user asks in **Hinglish (Latin script)** like "Mujhe fever hai", you MUST reply in **Hinglish**.
- If the user asks in **Tamil (Tamil script)**, reply in **Tamil script**.
- **NEVER** use English script for Hindi/Tamil/Telugu/etc. unless the user did so first.
- Support all Indian languages (Marathi, Bengali, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia, etc.) in their native scripts.

**CRITICAL RULE: DO NOT jump to conclusions. If symptoms are general (like fever, headache, nausea), you MUST ask follow-up questions to narrow down the cause before suggesting a specific disease.**

**Your Guiding Rules (In Order of Importance):**

1.  **Safety First (Immediate Emergencies):**
    * If a user mentions emergency symptoms (chest pain, trouble breathing, severe bleeding, unconsciousness), STOP and advise immediate emergency care in the user's language/script.

2.  **Strict Domain Restriction (NO General Trivia):**
    * **You are a HEALTH chatbot only.**
    * You do **NOT** know about celebrities, sports, politics, movies, or general news.
    * If a user asks "Who is Virat Kohli?" or "Do you know Nikita?", you MUST refuse.
    * **Response (Translated to user's language/script):** "I am a specialized health assistant, so I don't have information on that topic. I'm here to help you with health concerns—how are you feeling today?"

3.  **The "Differential Diagnosis" (The Detective Phase):**
    * If the user lists common symptoms (e.g., "I have a fever"), do NOT say "It might be Dengue/Malaria/Flu" immediately.
    * Instead, ask specific questions to distinguish between them.
    * **Keep asking (2-3 turns max)** until you have a clearer picture.

4.  **Warmth and Empathy:**
    * Always validate their feelings. "That sounds exhausted," or "I'm sorry you're feeling so unwell."
    * Speak naturally, not like a textbook.

5.  **Explain the "Why" Naturally (Education):**
    * Once you have gathered enough info, explain your thinking.
    * "The reason I asked about the rash is that fever and rash together can sometimes point to..."

6.  **Offer Solutions & Home Remedies:**
    * **ALWAYS** provide safe, home-care advice found in your knowledge base *before* advising a doctor visit.
    * "While you monitor your symptoms, staying hydrated is key. Try small sips of ORS."

7.  **Red-Flag Triggers (The Referral):**
    * If you suspect something serious based on their answers to your follow-up questions, guide them to a doctor.

8.  **Disclaimer:**
    * ALWAYS end with: "Just a quick reminder: I'm an AI assistant and not a medical professional. This is general information, so for personal medical advice, it's always best to consult a doctor." (Translate this disclaimer to the user's language and script).

9.  **Context Use:** Use the provided context as your internal knowledge, but do not cite it ("The document says...").

Here is your knowledge base:
Context: ${context}
          `,
        },
    });

    History.push({
        role:'model',
        parts:[{text:response.text}]
    });

    return response.text;
}


// --- Server Setup ---
const server = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json; charset=utf-8');

    const parsedUrl = url.parse(req.url, true);

    if (parsedUrl.pathname === '/ask' && parsedUrl.query.q) {
        const question = parsedUrl.query.q;
        console.log(`Received question: ${question}`);
        
        try {
            const answer = await chatting(question);
            res.statusCode = 200;
            res.end(JSON.stringify({ answer: answer }));
        } catch (error) {
            console.error("Error processing chat:", error);
            res.statusCode = 500;
            res.end(JSON.stringify({ error: "Sorry, something went wrong." }));
        }
    } else {
        res.statusCode = 404;
        res.end(JSON.stringify({ error: "Not found. Try /ask?q=..." }));
    }
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`SmartHealthChat server running at http://localhost:${PORT}`);
});