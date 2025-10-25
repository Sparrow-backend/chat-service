from pathlib import Path 
from datetime import datetime
from typing_extensions import Annotated, List, Literal
import os
import logging
import requests
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg

from src.llms.groqllm import GroqLLM

# Configure logging
logger = logging.getLogger(__name__)

# Configuration for MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", None)
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "parcel_tracking")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "parcels")
MONGODB_TIMEOUT = int(os.getenv("MONGODB_TIMEOUT", "5000"))  # milliseconds

# Configuration for ETA API
ETA_API_BASE_URL = os.getenv("ETA_API_BASE_URL", "http://localhost:8000")
ETA_API_TIMEOUT = int(os.getenv("ETA_API_TIMEOUT", "10"))  # seconds

# MongoDB client singleton
_mongo_client = None

def get_mongo_client():
    """Get or create MongoDB client singleton"""
    global _mongo_client
    
    if _mongo_client is None:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable is not set")
        
        try:
            _mongo_client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=MONGODB_TIMEOUT,
                connectTimeoutMS=MONGODB_TIMEOUT
            )
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    return _mongo_client

def get_today_str() -> str:
    """Get current data in a human-readable format."""
    return datetime.now().strftime("%a %b %d, %Y")

groq = GroqLLM()

@tool 
def think_tool(reflection:str) -> str:
    """
    Tool for strategic reflection on execution progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in customer query execution workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing execution gaps: What specific execution am I still missing?
    - Before concluding execution: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial execution or information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue execution or provide my output?

    Args: 
        reflection: Your detailed reflection on the execution progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


@tool(description="Track parcel based on tracking number")
def track_package(tracking_number: str) -> str:
    """
    Tool for tracking customer packages/parcels from MongoDB database.

    This tool retrieves real-time parcel tracking information from the MongoDB database.
    It fetches details such as current status, location, delivery ETA, sender/recipient info,
    and tracking history.

    Args:
        tracking_number(str): The unique tracking number of the parcel
    
    Returns:
        A string describing the parcel status, location, history, and other relevant details
    """
    logger.info(f"Tracking parcel: {tracking_number}")
    
    try:
        # Check if MongoDB is configured
        if not MONGODB_URI:
            error_msg = "MongoDB is not configured. Please set MONGODB_URI environment variable."
            logger.error(error_msg)
            return error_msg
        
        # Get MongoDB client and collection
        client = get_mongo_client()
        db = client[MONGODB_DATABASE]
        collection = db[MONGODB_COLLECTION]
        
        # Query for the tracking number
        logger.debug(f"Querying MongoDB for tracking_number: {tracking_number}")
        parcel = collection.find_one({"tracking_number": tracking_number})
        
        if not parcel:
            # Try case-insensitive search
            parcel = collection.find_one(
                {"tracking_number": {"$regex": f"^{tracking_number}$", "$options": "i"}}
            )
        
        if not parcel:
            logger.warning(f"Tracking number not found: {tracking_number}")
            return f"Tracking number '{tracking_number}' not found in the system. Please verify the tracking number and try again."
        
        # Format the response with all available information
        response_parts = []
        response_parts.append(f"📦 Parcel Tracking Information for {tracking_number}")
        response_parts.append("=" * 50)
        
        # Basic Information
        if parcel.get("status"):
            response_parts.append(f"\n🔹 Status: {parcel['status']}")
        
        if parcel.get("current_location"):
            response_parts.append(f"🔹 Current Location: {parcel['current_location']}")
        
        # Delivery Information
        if parcel.get("estimated_delivery"):
            response_parts.append(f"🔹 Estimated Delivery: {parcel['estimated_delivery']}")
        
        if parcel.get("actual_delivery_date"):
            response_parts.append(f"🔹 Delivered On: {parcel['actual_delivery_date']}")
        
        # Sender and Recipient Information
        if parcel.get("sender"):
            sender = parcel["sender"]
            if isinstance(sender, dict):
                sender_info = f"{sender.get('name', 'N/A')}"
                if sender.get('address'):
                    sender_info += f" ({sender['address']})"
                response_parts.append(f"\n🔹 Sender: {sender_info}")
            else:
                response_parts.append(f"\n🔹 Sender: {sender}")
        
        if parcel.get("recipient"):
            recipient = parcel["recipient"]
            if isinstance(recipient, dict):
                recipient_info = f"{recipient.get('name', 'N/A')}"
                if recipient.get('address'):
                    recipient_info += f" ({recipient['address']})"
                response_parts.append(f"🔹 Recipient: {recipient_info}")
            else:
                response_parts.append(f"🔹 Recipient: {recipient}")
        
        # Package Details
        if parcel.get("weight"):
            response_parts.append(f"\n🔹 Weight: {parcel['weight']}")
        
        if parcel.get("dimensions"):
            response_parts.append(f"🔹 Dimensions: {parcel['dimensions']}")
        
        if parcel.get("description"):
            response_parts.append(f"🔹 Description: {parcel['description']}")
        
        # Courier Information
        if parcel.get("courier_name"):
            response_parts.append(f"\n🔹 Courier: {parcel['courier_name']}")
        
        if parcel.get("vehicle_type"):
            response_parts.append(f"🔹 Vehicle Type: {parcel['vehicle_type']}")
        
        # Tracking History
        if parcel.get("tracking_history"):
            history = parcel["tracking_history"]
            if isinstance(history, list) and len(history) > 0:
                response_parts.append(f"\n📍 Tracking History:")
                for idx, event in enumerate(history[-5:], 1):  # Show last 5 events
                    if isinstance(event, dict):
                        timestamp = event.get('timestamp', 'N/A')
                        location = event.get('location', 'N/A')
                        status = event.get('status', 'N/A')
                        response_parts.append(f"  {idx}. [{timestamp}] {location} - {status}")
                    else:
                        response_parts.append(f"  {idx}. {event}")
                
                if len(history) > 5:
                    response_parts.append(f"  ... and {len(history) - 5} more events")
        
        # Additional Notes
        if parcel.get("notes"):
            response_parts.append(f"\n📝 Notes: {parcel['notes']}")
        
        # Special Instructions
        if parcel.get("special_instructions"):
            response_parts.append(f"⚠️ Special Instructions: {parcel['special_instructions']}")
        
        # Shipping Method
        if parcel.get("shipping_method"):
            response_parts.append(f"\n🔹 Shipping Method: {parcel['shipping_method']}")
        
        # Cost Information
        if parcel.get("shipping_cost"):
            response_parts.append(f"🔹 Shipping Cost: {parcel['shipping_cost']}")
        
        response_text = "\n".join(response_parts)
        logger.info(f"Successfully retrieved tracking info for: {tracking_number}")
        return response_text
        
    except ConnectionFailure as e:
        error_msg = f"Failed to connect to MongoDB: {str(e)}. Please check your connection."
        logger.error(error_msg)
        return error_msg
        
    except OperationFailure as e:
        error_msg = f"MongoDB operation failed: {str(e)}. Please check your permissions."
        logger.error(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"Error tracking parcel '{tracking_number}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

@tool(description="Estimate delivery time for a parcel based on delivery parameters.")
def estimated_time_analysis(
    distance_km: float,
    courier_experience_yrs: float = 2.0,
    vehicle_type: str = "Scooter",
    weather: str = "Sunny",
    time_of_day: str = "Morning",
    traffic_level: str = "Medium"
) -> str:
    """
    Estimate delivery time for a parcel based on various delivery parameters.
    
    This tool makes an API call to the ETA prediction service which uses a trained ML model
    to predict delivery time based on distance, courier experience, vehicle type, weather,
    time of day, and traffic conditions.

    Args:
        distance_km: Distance in kilometers (must be positive, max 1000km)
        courier_experience_yrs: Courier experience in years (0-50, default: 2.0)
        vehicle_type: Type of vehicle - one of: 'Scooter', 'Pickup Truck', 'Motorcycle' (default: 'Scooter')
        weather: Weather condition - one of: 'Sunny', 'Rainy', 'Foggy', 'Snowy', 'Windy' (default: 'Sunny')
        time_of_day: Time of day - one of: 'Morning', 'Afternoon', 'Evening', 'Night' (default: 'Morning')
        traffic_level: Traffic level - one of: 'Low', 'Medium', 'High' (default: 'Medium')

    Returns:
        A string describing the estimated delivery time in minutes with the input parameters used.
        
    Example:
        estimated_time_analysis(distance_km=15.5, vehicle_type="Motorcycle", traffic_level="High")
    """
    logger.info(f"Estimating delivery time: distance={distance_km}km, vehicle={vehicle_type}, traffic={traffic_level}")
    
    try:
        # Prepare the request payload matching the API schema
        payload = {
            "Distance_km": distance_km,
            "Courier_Experience_yrs": courier_experience_yrs,
            "Vehicle_Type": vehicle_type,
            "Weather": weather,
            "Time_of_Day": time_of_day,
            "Traffic_Level": traffic_level
        }
        
        # Make API call to ETA prediction service
        api_url = f"{ETA_API_BASE_URL}/predict"
        logger.debug(f"Calling ETA API: {api_url} with payload: {payload}")
        
        response = requests.post(
            api_url,
            json=payload,
            timeout=ETA_API_TIMEOUT
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        predicted_time = result.get("predicted_delivery_time")
        
        if predicted_time is None:
            raise ValueError("No prediction returned from API")
        
        # Format response
        hours = int(predicted_time // 60)
        minutes = int(predicted_time % 60)
        
        time_str = ""
        if hours > 0:
            time_str += f"{hours} hour{'s' if hours > 1 else ''}"
        if minutes > 0:
            if hours > 0:
                time_str += " and "
            time_str += f"{minutes} minute{'s' if minutes != 1 else ''}"
        
        response_text = f"""Estimated Delivery Time Analysis:
- Predicted Time: {predicted_time:.1f} minutes ({time_str})
- Distance: {distance_km} km
- Vehicle Type: {vehicle_type}
- Courier Experience: {courier_experience_yrs} years
- Weather: {weather}
- Time of Day: {time_of_day}
- Traffic Level: {traffic_level}

This prediction is based on a trained machine learning model considering all the above factors."""
        
        logger.info(f"ETA prediction successful: {predicted_time:.1f} minutes")
        return response_text
        
    except requests.exceptions.ConnectionError:
        error_msg = f"Unable to connect to ETA prediction service at {ETA_API_BASE_URL}. Please ensure the service is running."
        logger.error(error_msg)
        return error_msg
        
    except requests.exceptions.Timeout:
        error_msg = f"ETA prediction service timed out after {ETA_API_TIMEOUT} seconds."
        logger.error(error_msg)
        return error_msg
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"ETA prediction service returned an error: {e.response.status_code} - {e.response.text}"
        logger.error(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"Error during ETA estimation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

@tool
def conduct_execution(execution_jobs: str) -> str:
    """
    Tool for delegating an execution task to a specialized sub-agent.
    """
    return f"Delegated execution job: {execution_jobs}"


@tool
def execution_complete() -> str:
    """
    Tool for indicating the execution process is complete.
    """
    return "Execution complete."