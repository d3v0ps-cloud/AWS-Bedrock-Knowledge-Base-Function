#!/usr/bin/env python3
"""
Test script for AWS Bedrock Knowledge Base Function
"""
import asyncio
import json
import logging
import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from aws_bedrock_kb_function import Pipe

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('kb_test')

def list_all_knowledge_bases():
    """List all knowledge bases in the account"""
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION')
    
    logger.info(f"Listing all knowledge bases in region: {region}")
    
    try:
        # Create a session with the provided credentials
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Create a Bedrock Agent client
        bedrock_agent = session.client('bedrock-agent')
        
        # List all knowledge bases
        response = bedrock_agent.list_knowledge_bases()
        
        if 'knowledgeBaseSummaries' in response and response['knowledgeBaseSummaries']:
            logger.info(f"Found {len(response['knowledgeBaseSummaries'])} knowledge bases:")
            for kb in response['knowledgeBaseSummaries']:
                logger.info(f"  - ID: {kb.get('knowledgeBaseId', 'N/A')}")
                logger.info(f"    Name: {kb.get('name', 'N/A')}")
                logger.info(f"    Status: {kb.get('status', 'N/A')}")
                logger.info(f"    Created: {kb.get('createdAt', 'N/A')}")
        else:
            logger.warning("No knowledge bases found in this account/region")
            
        return response
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {str(e)}")
        return None

def get_data_source_details(kb_id=None, data_source_id=None):
    """Get details about a specific data source"""
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION')
    kb_id = kb_id or os.getenv('KNOWLEDGE_BASE_ID')
    data_source_id = data_source_id or os.getenv('DATA_SOURCE_ID')
    
    logger.info(f"Getting details for data source {data_source_id} in knowledge base {kb_id}")
    
    try:
        # Create a session with the provided credentials
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Create a Bedrock Agent client
        bedrock_agent = session.client('bedrock-agent')
        
        # Try different API methods to get data source details
        try:
            # Try first method
            response = bedrock_agent.get_data_source(
                knowledgeBaseId=kb_id,
                dataSourceId=data_source_id
            )
            
            logger.info("Data Source Details:")
            logger.info(f"Name: {response.get('name', 'N/A')}")
            logger.info(f"Status: {response.get('status', 'N/A')}")
            logger.info(f"Created At: {response.get('createdAt', 'N/A')}")
            logger.info(f"Updated At: {response.get('updatedAt', 'N/A')}")
            
            # Check for data source configuration
            if 'dataSourceConfiguration' in response:
                config = response['dataSourceConfiguration']
                logger.info("Data Source Configuration:")
                
                if 's3Configuration' in config:
                    s3_config = config['s3Configuration']
                    logger.info(f"  S3 Bucket: {s3_config.get('bucketName', 'N/A')}")
                    logger.info(f"  S3 Prefix: {s3_config.get('bucketPrefix', 'N/A')}")
                    logger.info(f"  Inclusion Prefixes: {s3_config.get('inclusionPrefixes', [])}")
                    logger.info(f"  Inclusion Patterns: {s3_config.get('inclusionPatterns', [])}")
            
            return response
            
        except Exception as e1:
            logger.warning(f"First method failed: {str(e1)}")
            
            try:
                # Try alternative method
                response = bedrock_agent.get_knowledge_base_data_source(
                    knowledgeBaseId=kb_id,
                    dataSourceId=data_source_id
                )
                
                logger.info("Data Source Details (alternative method):")
                logger.info(f"Name: {response.get('name', 'N/A')}")
                logger.info(f"Status: {response.get('status', 'N/A')}")
                logger.info(f"Created At: {response.get('createdAt', 'N/A')}")
                logger.info(f"Updated At: {response.get('updatedAt', 'N/A')}")
                
                return response
                
            except Exception as e2:
                logger.error(f"Second method failed: {str(e2)}")
                return None
                
    except Exception as e:
        logger.error(f"Error getting data source details: {str(e)}")
        return None

def check_knowledge_base_details(kb_id=None):
    """Check details of the knowledge base"""
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION')
    kb_id = kb_id or os.getenv('KNOWLEDGE_BASE_ID')
    
    logger.info(f"Checking knowledge base details for KB ID: {kb_id}")
    
    try:
        # Create a session with the provided credentials
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Create a Bedrock Agent client
        bedrock_agent = session.client('bedrock-agent')
        
        try:
            # Get knowledge base details
            response = bedrock_agent.get_knowledge_base(
                knowledgeBaseId=kb_id
            )
            
            logger.info("Knowledge Base Details:")
            logger.info(f"Name: {response.get('name', 'N/A')}")
            logger.info(f"Status: {response.get('status', 'N/A')}")
            logger.info(f"Created At: {response.get('createdAt', 'N/A')}")
            logger.info(f"Updated At: {response.get('updatedAt', 'N/A')}")
        except Exception as e:
            logger.error(f"Error getting knowledge base details: {str(e)}")
        
        # Try to list data sources using different method names
        try:
            # Try the first method name
            data_sources = bedrock_agent.list_data_sources(
                knowledgeBaseId=kb_id
            )
            
            logger.info(f"Data Sources: {len(data_sources.get('dataSourceSummaries', []))}")
            for ds in data_sources.get('dataSourceSummaries', []):
                logger.info(f"  - {ds.get('name', 'N/A')} (Status: {ds.get('status', 'N/A')})")
                
        except Exception as e1:
            logger.warning(f"First data source listing method failed: {str(e1)}")
            
            try:
                # Try alternative method name
                data_sources = bedrock_agent.list_knowledge_base_data_sources(
                    knowledgeBaseId=kb_id
                )
                
                logger.info(f"Data Sources: {len(data_sources.get('knowledgeBaseDataSourceSummaries', []))}")
                for ds in data_sources.get('knowledgeBaseDataSourceSummaries', []):
                    logger.info(f"  - {ds.get('name', 'N/A')} (Status: {ds.get('status', 'N/A')})")
                    
            except Exception as e2:
                logger.error(f"Second data source listing method failed: {str(e2)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error checking knowledge base details: {str(e)}")
        return None

async def test_kb_query(query, debug=False):
    """Test the knowledge base query functionality"""
    if debug:
        # Enable detailed boto3 logging for debugging
        boto3.set_stream_logger('', logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Initializing pipe with query: '{query}'")
    pipe = Pipe()
    
    # Configure the pipe with your AWS credentials and settings from environment variables
    pipe.valves.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    pipe.valves.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    pipe.valves.aws_region = os.getenv('AWS_REGION')
    pipe.valves.knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID')
    
    # Optional: Configure additional parameters from environment variables
    pipe.valves.model_id = os.getenv('MODEL_ID', "anthropic.claude-3-sonnet-20240229-v1:0")
    pipe.valves.number_of_results = int(os.getenv('NUMBER_OF_RESULTS', 10))  # Increase number of results
    
    # Create a mock request body with the query
    body = {
        "messages": [
            {"role": "user", "content": query}
        ]
    }
    
    # Initialize clients directly to test knowledge base access
    logger.info("Initializing AWS clients")
    pipe._initialize_clients()
    
    # Test direct knowledge base query
    try:
        logger.info("Directly querying knowledge base to debug")
        direct_response = pipe.bedrock_agent_client.retrieve(
            knowledgeBaseId=pipe.valves.knowledge_base_id,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': pipe.valves.number_of_results
                }
            }
        )
        
        # Print raw response for debugging
        logger.info("Raw knowledge base response:")
        logger.info(json.dumps(direct_response, default=str, indent=2))
        
        # Check if there are any results
        retrieved_results = direct_response.get('retrievalResults', [])
        if not retrieved_results:
            logger.warning("No results found in direct knowledge base query")
        else:
            logger.info(f"Found {len(retrieved_results)} results in direct query")
            for i, result in enumerate(retrieved_results, 1):
                if 'content' in result and 'text' in result['content']:
                    logger.info(f"Result {i}: {result['content']['text'][:100]}...")
    except Exception as e:
        logger.error(f"Error in direct knowledge base query: {str(e)}")
    
    # Call the pipe function
    logger.info("Calling pipe function")
    response = await pipe.pipe(body)
    
    # Print the response
    if isinstance(response, dict) and "error" in response:
        logger.error(f"Error: {response['error']}")
    else:
        logger.info("Response from Knowledge Base:")
        logger.info(response)
    
    # Print the updated messages in the body
    logger.info("\nUpdated messages:")
    logger.info(json.dumps(body["messages"], indent=2))
    
    return response

async def run_movie_queries():
    """Run a series of movie-related queries to test the knowledge base"""
    movie_queries = [
        "list all star trek movies",
        "Tell me about The Godfather movie",
        "Who directed Inception?",
        "What is the plot of Pulp Fiction?",
        "List some popular action movies",
        "Who starred in The Shawshank Redemption?",
        "movie"  # Simple query to match any movie content
    ]
    
    results = []
    for query in movie_queries:
        print("\n" + "="*80)
        print(f"Testing query: '{query}'")
        print("="*80)
        result = await test_kb_query(query)
        results.append((query, result))
        print("\n")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Get configuration from environment variables
    kb_id = os.getenv('KNOWLEDGE_BASE_ID')
    data_source_id = os.getenv('DATA_SOURCE_ID')
    
    # Check command line arguments
    if "--list-kbs" in sys.argv:
        print("Listing all knowledge bases in the account...")
        list_all_knowledge_bases()
    elif "--check-kb" in sys.argv:
        print("Checking knowledge base details...")
        check_knowledge_base_details()
    elif "--check-ds" in sys.argv:
        print("Checking data source details...")
        get_data_source_details()
    elif len(sys.argv) > 1 and sys.argv[1] not in ["--debug", "--check-kb", "--list-kbs", "--check-ds"]:
        # Run with specific query
        query = sys.argv[1]
        debug_mode = "--debug" in sys.argv
        print(f"Testing AWS Bedrock Knowledge Base with query: '{query}'")
        asyncio.run(test_kb_query(query, debug=debug_mode))
    else:
        # First list all knowledge bases
        print("Listing all knowledge bases in the account...")
        list_all_knowledge_bases()
        
        # Then check KB details
        print("\nChecking knowledge base details...")
        check_knowledge_base_details()
        
        # Then check data source details
        print("\nChecking data source details...")
        get_data_source_details()
        
        # Then run a series of movie-related queries
        print("\nRunning multiple movie-related queries to test knowledge base")
        asyncio.run(run_movie_queries())