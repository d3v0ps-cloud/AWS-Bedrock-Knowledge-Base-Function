"""
title: AWS Bedrock Knowledge Base Function
author: Aaron Bolton
version: 0.1.0
description: Integration with AWS Bedrock Knowledge Base for OpenWebUI
This module defines a Pipe class that utilizes AWS Bedrock Knowledge Base for retrieving information
from your documents and providing AI-generated responses.
"""
from typing import Optional, Callable, Awaitable, List, Dict, Any
from pydantic import BaseModel, Field
import os
import time
import json
import boto3
from botocore.exceptions import ClientError

def extract_event_info(event_emitter) -> tuple[Optional[str], Optional[str]]:
    """Extract chat_id and message_id from event emitter closure"""
    if not event_emitter or not event_emitter.__closure__:
        return None, None
    for cell in event_emitter.__closure__:
        if isinstance(request_info := cell.cell_contents, dict):
            chat_id = request_info.get("chat_id")
            message_id = request_info.get("message_id")
            return chat_id, message_id
    return None, None

class Pipe:
    class Valves(BaseModel):
        aws_access_key_id: str = Field(
            default="", description="AWS Access Key ID"
        )
        aws_secret_access_key: str = Field(
            default="", description="AWS Secret Access Key"
        )
        aws_session_token: str = Field(
            default="", description="AWS Session Token (optional, for temporary credentials)"
        )
        aws_region: str = Field(
            default="eu-west-1", description="AWS Region"
        )
        knowledge_base_id: str = Field(
            default="", description="AWS Bedrock Knowledge Base ID"
        )
        model_id: str = Field(
            default="anthropic.claude-3-sonnet-20240229-v1:0",
            description="Model ID to use for retrieval"
        )
        max_tokens: int = Field(
            default=4096, description="Maximum number of tokens in the response"
        )
        temperature: float = Field(
            default=0.7, description="Temperature for model generation"
        )
        top_p: float = Field(
            default=0.9, description="Top-p sampling parameter"
        )
        number_of_results: int = Field(
            default=5, description="Number of knowledge base results to retrieve"
        )
        use_conversation_history: bool = Field(
            default=True, description="Whether to include conversation history for context"
        )
        max_history_messages: int = Field(
            default=10, description="Maximum number of previous messages to include in history"
        )
        emit_interval: float = Field(
            default=2.0, description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "aws_bedrock_kb"
        self.name = "AWS Bedrock Knowledge Base"
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.bedrock_client = None
        self.bedrock_agent_client = None

    def _initialize_clients(self):
        """Initialize AWS Bedrock clients with credentials"""
        if not self.bedrock_client or not self.bedrock_agent_client:
            session_kwargs = {
                'aws_access_key_id': self.valves.aws_access_key_id,
                'aws_secret_access_key': self.valves.aws_secret_access_key,
                'region_name': self.valves.aws_region
            }
            
            # Add session token if provided
            if self.valves.aws_session_token:
                session_kwargs['aws_session_token'] = self.valves.aws_session_token
                
            session = boto3.Session(**session_kwargs)
            
            try:
                self.bedrock_client = session.client('bedrock-runtime')
                self.bedrock_agent_client = session.client('bedrock-agent-runtime')
            except Exception as e:
                raise Exception(f"Failed to initialize AWS clients: {str(e)}")

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        """Emit status updates to the UI"""
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (
                current_time - self.last_emit_time >= self.valves.emit_interval or done
            )
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    def _format_conversation_history(self, messages):
        """Format conversation history for context"""
        if not self.valves.use_conversation_history:
            return ""
            
        # Get the last N messages (excluding the current question)
        history_messages = messages[:-1]
        if len(history_messages) > self.valves.max_history_messages:
            history_messages = history_messages[-self.valves.max_history_messages:]
            
        if not history_messages:
            return ""
            
        formatted_history = "Previous conversation:\n\n"
        for msg in history_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted_history += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_history += f"Assistant: {content}\n\n"
                
        return formatted_history + "\n"

    async def query_knowledge_base(self, query: str, chat_id: str, conversation_history="") -> str:
        """Query the AWS Bedrock Knowledge Base"""
        self._initialize_clients()
        
        try:
            # Query the knowledge base
            response = self.bedrock_agent_client.retrieve(
                knowledgeBaseId=self.valves.knowledge_base_id,
                retrievalQuery={
                    'text': query
                },
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': self.valves.number_of_results
                    }
                }
            )
            
            # Extract retrieved passages
            retrieved_results = response.get('retrievalResults', [])
            context = ""
            
            # Add source information to each result
            for i, result in enumerate(retrieved_results, 1):
                if 'content' in result and 'text' in result['content']:
                    content = result['content']['text']
                    source = ""
                    if 'location' in result:
                        source = f" (Source: {result['location'].get('s3Location', {}).get('uri', 'Unknown')})"
                    context += f"[Document {i}{source}]\n{content}\n\n"
            
            # If no results were found
            if not context:
                return "I couldn't find any relevant information in the knowledge base."
            
            # Generate a response using the retrieved context and conversation history
            prompt = f"""
            {conversation_history}
            
            The following information was retrieved from a knowledge base:
            
            {context}
            
            Based on this information, please answer the following question:
            {query}
            
            If the information doesn't contain a clear answer, please say so.
            """
            
            # Call Bedrock model to generate a response
            try:
                model_response = self.bedrock_client.invoke_model(
                    modelId=self.valves.model_id,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.valves.max_tokens,
                        "temperature": self.valves.temperature,
                        "top_p": self.valves.top_p,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                )
                
                # Parse the response
                response_body = json.loads(model_response['body'].read())
                return response_body['content'][0]['text']
                
            except ClientError as e:
                if "AccessDeniedException" in str(e):
                    return "Error: Access denied to AWS Bedrock. Please check your AWS credentials and permissions."
                elif "ValidationException" in str(e):
                    return "Error: Invalid request to AWS Bedrock. Please check your model ID and parameters."
                elif "ThrottlingException" in str(e):
                    return "Error: AWS Bedrock request was throttled. Please try again later."
                elif "ServiceQuotaExceededException" in str(e):
                    return "Error: AWS Bedrock service quota exceeded. Please try again later or request a quota increase."
                else:
                    return f"AWS Bedrock error: {str(e)}"
                    
        except ClientError as e:
            if "ResourceNotFoundException" in str(e):
                return f"Error: Knowledge Base ID '{self.valves.knowledge_base_id}' not found. Please check your Knowledge Base ID."
            elif "AccessDeniedException" in str(e):
                return "Error: Access denied to AWS Bedrock Knowledge Base. Please check your AWS credentials and permissions."
            elif "ValidationException" in str(e):
                return "Error: Invalid request to AWS Bedrock Knowledge Base. Please check your parameters."
            else:
                return f"AWS Bedrock Knowledge Base error: {str(e)}"
        except Exception as e:
            error_message = f"Error querying knowledge base: {str(e)}"
            return error_message

    async def pipe(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[dict]:
        """Main pipe function that processes the input and returns a response"""
        await self.emit_status(
            __event_emitter__, "info", "Querying AWS Bedrock Knowledge Base...", False
        )
        
        chat_id, _ = extract_event_info(__event_emitter__)
        messages = body.get("messages", [])
        
        # Verify a message is available
        if messages:
            question = messages[-1]["content"]
            try:
                # Check if AWS credentials are provided
                if not self.valves.aws_access_key_id or not self.valves.aws_secret_access_key:
                    error_message = "AWS credentials are not configured. Please set aws_access_key_id and aws_secret_access_key in the function settings."
                    await self.emit_status(__event_emitter__, "error", error_message, True)
                    body["messages"].append({"role": "assistant", "content": error_message})
                    return {"error": error_message}
                    
                # Check if Knowledge Base ID is provided
                if not self.valves.knowledge_base_id:
                    error_message = "Knowledge Base ID is not configured. Please set knowledge_base_id in the function settings."
                    await self.emit_status(__event_emitter__, "error", error_message, True)
                    body["messages"].append({"role": "assistant", "content": error_message})
                    return {"error": error_message}
                
                # Format conversation history if enabled
                conversation_history = ""
                if self.valves.use_conversation_history:
                    await self.emit_status(
                        __event_emitter__, "info", "Processing conversation history...", False
                    )
                    conversation_history = self._format_conversation_history(messages)
                
                # Query the knowledge base
                await self.emit_status(
                    __event_emitter__, "info", "Retrieving information from Knowledge Base...", False
                )
                
                kb_response = await self.query_knowledge_base(question, chat_id, conversation_history)
                
                # Set assistant message with response
                body["messages"].append({"role": "assistant", "content": kb_response})
                
                await self.emit_status(__event_emitter__, "info", "Complete", True)
                return kb_response
                
            except Exception as e:
                error_message = f"Error during knowledge base query: {str(e)}"
                await self.emit_status(
                    __event_emitter__,
                    "error",
                    error_message,
                    True,
                )
                body["messages"].append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
                return {"error": error_message}
        # If no message is available alert user
        else:
            error_message = "No messages found in the request body"
            await self.emit_status(
                __event_emitter__,
                "error",
                error_message,
                True,
            )
            body["messages"].append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )
            return {"error": error_message}