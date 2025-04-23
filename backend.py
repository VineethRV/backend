#! ../myenv/bin/python
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import requests
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, timedelta
import functools
import time

app = FastAPI(title="Aviation Weather API for Flight Routes", 
              description="API for retrieving comprehensive aviation weather data along a flight route")

# Simple in-memory cache with TTL
cache = {}
CACHE_TTL = 300  # 5 minutes

def timed_cache(seconds=CACHE_TTL):
    """Simple time-based cache decorator"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            # Check if we have a valid cached response
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < seconds:
                    return result
            
            # Get fresh result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        return wrapper
    return decorator

class WeatherService:
    """Service for fetching and processing aviation weather data"""
    
    @staticmethod
    def parse_route(route_str: str) -> List[Dict[str, Any]]:
        """Parse a comma-separated route string into waypoints with ICAO and altitude"""
        if not route_str:
            return []
            
        parts = route_str.split(',')
        if len(parts) % 2 != 0:
            raise HTTPException(status_code=400, detail="Route must contain pairs of ICAO code and altitude")
            
        waypoints = []
        for i in range(0, len(parts), 2):
            try:
                icao = parts[i].strip().upper()
                altitude = int(parts[i+1].strip())
                
                # Validate ICAO format (typically 4 alphanumeric chars)
                if not re.match(r'^[A-Z0-9]{3,5}$', icao):
                    raise HTTPException(status_code=400, detail=f"Invalid ICAO code format: {icao}")
                
                waypoints.append({
                    "icao": icao,
                    "altitude": altitude
                })
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid altitude value: {parts[i+1]}")
                
        return waypoints
    
    @staticmethod
    @timed_cache(seconds=300)
    def fetch_metar(icao: str) -> str:
        """Fetch METAR data for a given ICAO code from aviationweather.gov"""
        url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=raw"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Extract the raw METAR from the response
            metar_data = response.text.strip()
            if not metar_data:
                return None
                
            return metar_data
        except requests.RequestException as e:
            return None
    
    @staticmethod
    @timed_cache(seconds=600)
    def fetch_taf(icao: str) -> str:
        """Fetch TAF data for a given ICAO code from aviationweather.gov"""
        url = f"https://aviationweather.gov/api/data/taf?ids={icao}&format=raw"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Extract the raw TAF from the response
            taf_data = response.text.strip()
            if not taf_data:
                return None
                
            return taf_data
        except requests.RequestException as e:
            return None
    
    @staticmethod
    @timed_cache(seconds=600)
    def fetch_pireps(icao: str) -> List[str]:
        """Fetch PIREPs for a location (within 50nm radius) from aviationweather.gov"""
        url = f"https://aviationweather.gov/api/data/pirep?station={icao}&format=raw&distance=50"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # The response should contain PIREPs separated by newlines
            pirep_data = response.text.strip()
            if not pirep_data:
                return []
            
            # Process the text to extract proper PIREP reports
            pireps = []
            # PIREPs typically start with UA or UUA
            pirep_matches = re.findall(r'(?:^|\n)((?:UA|UUA)[^\n]+)', pirep_data)
            
            if pirep_matches:
                pireps = [pirep.strip() for pirep in pirep_matches]
            
            # If no proper PIREPs found and we have data, check if we need to parse it differently
            if not pireps and pirep_data:
                # Try to identify if we have PIREPs in a different format
                if re.search(r'OV|FL|TP|TB|IC|RM|WX', pirep_data):
                    # This looks like PIREP data - treat the whole response as one PIREP
                    pireps = [pirep_data]
                    
            return pireps
        except requests.RequestException as e:
            return []
    
    @staticmethod
    @timed_cache(seconds=900)
    def fetch_sigmets(icao: str) -> List[str]:
        """Fetch SIGMETs that might affect the location"""
        url = f"https://aviationweather.gov/api/data/sigmet?stations={icao}&format=raw"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # SIGMETS should be separated by clear boundaries
            sigmet_data = response.text.strip()
            if not sigmet_data:
                return []
            
            # Parse out individual SIGMET messages
            # SIGMETs typically start with WSXX (WS for US, WC for Canada, etc.)
            sigmet_matches = re.findall(r'(?:^|\n)([WA-Z]{2}[A-Z]{2}\d{2}[^\n]+(?:\n[^\n]+)*)', sigmet_data)
            
            if sigmet_matches:
                sigmets = [sigmet.strip() for sigmet in sigmet_matches]
                return sigmets
            
            # If no proper SIGMETs found with standard format but we have data
            if sigmet_data and "SIGMET" in sigmet_data:
                # This looks like it might be SIGMET data in a different format
                # Try to split on double newlines or other delimiters
                potential_sigmets = re.split(r'\n\s*\n', sigmet_data)
                sigmets = [s.strip() for s in potential_sigmets if s.strip()]
                return sigmets
                
            return []
        except requests.RequestException as e:
            return []
    
    @staticmethod
    def decode_metar(metar: str) -> Dict[str, Any]:
        """Parse METAR data into structured format"""
        if not metar:
            return {
                "decoded": {
                    "raw": "",
                    "conditions": "Unknown",
                    "visibility": "Unknown",
                    "wind": "Unknown",
                    "temperature": "Unknown",
                    "dewpoint": "Unknown",
                    "clouds": "Unknown"
                },
                "summary": "No METAR data available"
            }
            
        # Extract key components using regex patterns
        decoded = {
            "raw": metar,
            "conditions": "Unknown",
            "visibility": "Unknown",
            "wind": "Unknown",
            "temperature": "Unknown",
            "dewpoint": "Unknown",
            "clouds": "Unknown"
        }
        
        # Wind (direction, speed)
        wind_match = re.search(r'\b(\d{3})(\d{2})(G\d{2})?KT\b', metar)
        if wind_match:
            direction = wind_match.group(1)
            speed = wind_match.group(2)
            gust = wind_match.group(3)[1:] if wind_match.group(3) else None
            
            wind_info = f"{direction}° at {speed} knots"
            if gust:
                wind_info += f" (gusts {gust} knots)"
            decoded["wind"] = wind_info
        
        # Visibility
        vis_match = re.search(r'\b(\d+)SM\b', metar)
        if vis_match:
            decoded["visibility"] = f"{vis_match.group(1)}SM"
        
        # Temperature/Dewpoint
        temp_match = re.search(r'\b(\d{2})/(\d{2})\b', metar)
        if temp_match:
            temp = int(temp_match.group(1))
            dew = int(temp_match.group(2))
            decoded["temperature"] = f"{temp}°C"
            decoded["dewpoint"] = f"{dew}°C"
        
        # Cloud layers
        cloud_pattern = r'\b(SKC|CLR|FEW|SCT|BKN|OVC)(\d{3})?\b'
        cloud_matches = re.findall(cloud_pattern, metar)
        
        if cloud_matches:
            cloud_layers = []
            for match in cloud_matches:
                cover = match[0]
                height = match[1]
                
                cover_terms = {
                    "SKC": "sky clear",
                    "CLR": "clear",
                    "FEW": "few clouds",
                    "SCT": "scattered clouds",
                    "BKN": "broken clouds",
                    "OVC": "overcast"
                }
                
                cover_text = cover_terms.get(cover, cover)
                if height:
                    altitude = int(height) * 100
                    cloud_layers.append(f"{cover_text} at {altitude} feet")
                else:
                    cloud_layers.append(cover_text)
                    
            decoded["clouds"] = ", ".join(cloud_layers)
        
        # Weather conditions
        wx_patterns = [
            (r'\bTS\b', "thunderstorm"),
            (r'\bRA\b', "rain"),
            (r'\bDZ\b', "drizzle"),
            (r'\bSN\b', "snow"),
            (r'\bFG\b', "fog"),
            (r'\bBR\b', "mist"),
            (r'\bHZ\b', "haze"),
            (r'\bSA\b', "sand"),
            (r'\bDU\b', "dust"),
            (r'\bSQ\b', "squall"),
            (r'\bFC\b', "funnel cloud"),
            (r'\b\+\w+\b', "heavy"),
            (r'\b-\w+\b', "light")
        ]
        
        conditions = []
        for pattern, description in wx_patterns:
            if re.search(pattern, metar):
                conditions.append(description)
        
        if not conditions and ("CLR" in metar or "SKC" in metar):
            conditions = ["clear"]
        
        decoded["conditions"] = ", ".join(conditions) if conditions else "no significant weather"
        
        # Create human-readable summary
        summary = f"{decoded['clouds']}, visibility {decoded['visibility']}, "
        summary += f"wind {decoded['wind']}, temperature {decoded['temperature']}"
        if decoded['conditions'] != "no significant weather":
            summary += f", {decoded['conditions']}"
        
        return {
            "decoded": decoded,
            "summary": summary
        }
    
    @staticmethod
    def decode_taf(taf: str) -> Dict[str, Any]:
        """Parse TAF data into a human-readable summary"""
        if not taf:
            return {
                "summary": "No TAF data available"
            }
            
        # Create a simple summary - for TAF we'll focus on forecast period and key weather changes
        forecast_periods = []
        
        # Extract forecast time periods and conditions
        period_matches = re.findall(r'\b(FM|BECMG|TEMPO)\b\s?(\d{6})?', taf)
        
        if period_matches:
            for i, (change_type, time_str) in enumerate(period_matches):
                if time_str:
                    try:
                        hour = time_str[0:2]
                        day = time_str[2:4]
                        time_desc = f"{day}th at {hour}:00Z"
                    except:
                        time_desc = time_str
                else:
                    time_desc = "unknown time"
                
                change_types = {
                    "FM": "From",
                    "BECMG": "Becoming",
                    "TEMPO": "Temporarily"
                }
                
                # Find the next period to extract text between
                next_idx = taf.find(period_matches[i+1][0]) if i+1 < len(period_matches) else len(taf)
                current_idx = taf.find(f"{change_type}{time_str}" if time_str else change_type)
                
                if current_idx >= 0 and next_idx > current_idx:
                    period_text = taf[current_idx:next_idx].strip()
                    desc = f"{change_types.get(change_type, change_type)} {time_desc}: {period_text}"
                    forecast_periods.append(desc)
        
        # If no periods found, try to extract the main forecast period
        if not forecast_periods:
            # Look for the TAF header pattern
            taf_match = re.search(r'TAF\s+(?:AMD\s+)?([A-Z0-9]{4})\s+(\d{6}Z)\s+(\d{4}/\d{4})', taf)
            if taf_match:
                station = taf_match.group(1)
                issue_time = taf_match.group(2)
                valid_period = taf_match.group(3)
                forecast_periods.append(f"TAF for {station} issued {issue_time}, valid {valid_period}")
            else:
                # Just use the whole TAF
                forecast_periods = [taf]
        
        # Create summary
        summary = "Forecast: " + "; ".join(forecast_periods)
        
        return {
            "summary": summary[:250] + "..." if len(summary) > 250 else summary  # Limit length
        }
    
    @staticmethod
    def decode_pireps(pireps: List[str]) -> List[Dict[str, Any]]:
        """Parse PIREPs into human-readable summaries"""
        if not pireps:
            return []
            
        decoded_pireps = []
        
        for pirep in pireps:
            # Skip if the pirep is just a single character or obviously not a valid PIREP
            if len(pirep) < 5:
                continue
                
            # Extract basic info - aircraft type, altitude, location
            aircraft_match = re.search(r'/TP\s+(\w+)', pirep)
            altitude_match = re.search(r'/FL(\d{3})|/(\d{3})(?!\d)', pirep)
            location_match = re.search(r'OV\s+([A-Z0-9]+)', pirep)
            time_match = re.search(r'\b(\d{2})(\d{2})Z\b', pirep)
            wx_match = re.search(r'WX\s+([^/]+)(?:/|$)', pirep)
            turbulence_match = re.search(r'TB\s+([^/]+)(?:/|$)', pirep)
            icing_match = re.search(r'IC\s+([^/]+)(?:/|$)', pirep)
            
            summary_parts = []
            
            # Aircraft
            if aircraft_match:
                summary_parts.append(f"Aircraft: {aircraft_match.group(1)}")
            
            # Altitude
            altitude = None
            if altitude_match:
                if altitude_match.group(1):  # Flight level
                    altitude = int(altitude_match.group(1)) * 100
                elif altitude_match.group(2):  # Direct altitude
                    altitude = int(altitude_match.group(2)) * 100
                summary_parts.append(f"Altitude: {altitude} ft")
            
            # Location
            if location_match:
                summary_parts.append(f"Over: {location_match.group(1)}")
            
            # Time
            if time_match:
                summary_parts.append(f"Time: {time_match.group(1)}:{time_match.group(2)}Z")
            
            # Weather
            if wx_match:
                summary_parts.append(f"Weather: {wx_match.group(1)}")
            
            # Turbulence
            if turbulence_match:
                summary_parts.append(f"Turbulence: {turbulence_match.group(1)}")
            
            # Icing
            if icing_match:
                summary_parts.append(f"Icing: {icing_match.group(1)}")
                
            # If we couldn't extract any specific parts, use some general text
            if not summary_parts:
                # Check if this looks like a valid PIREP using basic patterns
                if re.search(r'UA|UUA|OV|FL|TP|TB|IC|RM|WX', pirep):
                    summary_parts = ["Pilot report available - see raw data for details"]
                else:
                    # This doesn't look like a PIREP at all, skip it
                    continue
            
            decoded_pireps.append({
                "raw": pirep,
                "summary": ", ".join(summary_parts),
                "altitude": altitude
            })
        
        return decoded_pireps
    
    @staticmethod
    def decode_sigmets(sigmets: List[str]) -> List[Dict[str, Any]]:
        """Parse SIGMETs into human-readable summaries"""
        if not sigmets:
            return []
            
        decoded_sigmets = []
        
        for sigmet in sigmets:
            # Skip if the sigmet is just a single character or obviously not a valid SIGMET
            if len(sigmet) < 5:
                continue
                
            # Extract basic info from SIGMET
            valid_match = re.search(r'VALID\s+(\d{6}(?:/\d{6})?)', sigmet)
            area_match = re.search(r'AREA\s+([^-]+)', sigmet)
            hazard_match = re.search(r'(CONVECTIVE|TURB|ICE|IFR|MTN OBSCN|VA|DS|SS|TC)', sigmet)
            
            summary_parts = []
            
            # Valid period
            if valid_match:
                summary_parts.append(f"Valid: {valid_match.group(1)}")
            
            # Area
            if area_match:
                summary_parts.append(f"Area: {area_match.group(1).strip()}")
            
            # Hazard type
            if hazard_match:
                hazard_types = {
                    "CONVECTIVE": "Thunderstorms",
                    "TURB": "Turbulence",
                    "ICE": "Icing",
                    "IFR": "Instrument conditions",
                    "MTN OBSCN": "Mountains obscured",
                    "VA": "Volcanic ash",
                    "DS": "Dust storm",
                    "SS": "Sandstorm",
                    "TC": "Tropical cyclone"
                }
                hazard = hazard_types.get(hazard_match.group(1), hazard_match.group(1))
                summary_parts.append(f"Hazard: {hazard}")
            
            # Look for specific warnings about severity
            severity_match = re.search(r'(SEV|MOD|INTSFR|EXTM)', sigmet)
            if severity_match:
                severity_types = {
                    "SEV": "Severe",
                    "MOD": "Moderate",
                    "INTSFR": "Intense",
                    "EXTM": "Extreme"
                }
                severity = severity_types.get(severity_match.group(1), severity_match.group(1))
                summary_parts.append(f"Severity: {severity}")
                
            # If we couldn't extract any specific parts but it has SIGMET in it
            if not summary_parts and "SIGMET" in sigmet:
                summary_parts = ["SIGMET available - see raw data for details"]
            elif not summary_parts:
                # This doesn't look like a SIGMET at all, skip it
                continue
            
            decoded_sigmets.append({
                "raw": sigmet,
                "summary": ", ".join(summary_parts)
            })
        
        return decoded_sigmets
    
    @staticmethod
    def classify_weather(decoded_metar: Dict[str, Any], pireps: List[Dict[str, Any]], sigmets: List[Dict[str, Any]]) -> str:
        """Classify weather conditions as VFR, Significant, or Severe based on all available data"""
        # Start with METAR-based classification
        decoded = decoded_metar.get("decoded", {})
        
        # Extract visibility (number only)
        visibility = decoded.get("visibility", "Unknown")
        vis_value = 0
        if "SM" in visibility:
            try:
                vis_value = float(visibility.replace("SM", "").strip())
            except ValueError:
                vis_value = 0
        
        # Extract wind speed
        wind = decoded.get("wind", "Unknown")
        wind_speed = 0
        wind_match = re.search(r'at (\d+) knots', wind)
        if wind_match:
            wind_speed = int(wind_match.group(1))
        
        # Check conditions from METAR
        conditions = decoded.get("conditions", "").lower()
        clouds = decoded.get("clouds", "").lower()
        
        # Default classification based on METAR
        if "thunderstorm" in conditions or vis_value < 3 or wind_speed > 25:
            classification = "Severe"
        elif ("rain" in conditions or "broken" in clouds or "overcast" in clouds or wind_speed > 15):
            classification = "Significant"
        elif vis_value >= 6 and "rain" not in conditions and "thunderstorm" not in conditions:
            classification = "VFR"
        else:
            classification = "Significant"  # Default to Significant if uncertain
        
        # Check PIREPs for severe conditions
        for pirep in pireps:
            summary = pirep.get("summary", "").lower()
            if any(severe in summary.lower() for severe in ["sev turb", "severe turb", "severe icing", "sev icing", "mod-sev", "heavy"]):
                classification = "Severe"
                break
            elif any(mod in summary.lower() for mod in ["moderate turb", "mod turb", "moderate icing", "mod icing"]):
                classification = max(classification, "Significant", key=lambda x: {"VFR": 0, "Significant": 1, "Severe": 2}.get(x, 0))
        
        # Check SIGMETs
        for sigmet in sigmets:
            if "thunderstorm" in sigmet.get("summary", "").lower() or "severe" in sigmet.get("summary", "").lower():
                classification = "Severe"
                break
            elif any(wx in sigmet.get("summary", "").lower() for wx in ["turbulence", "icing", "instrument", "obscured"]):
                classification = max(classification, "Significant", key=lambda x: {"VFR": 0, "Significant": 1, "Severe": 2}.get(x, 0))
        
        return classification

@app.get("/weather", response_class=JSONResponse)
async def get_route_weather(route: str = Query(..., description="Comma-separated list of ICAO codes and altitudes")):
    """
    Get comprehensive aviation weather data for a flight route
    
    - **route**: Comma-separated list of ICAO codes and altitudes in feet
      Example: KPHX,1500,KBXK,12000,KPSP,20000,KLAX,50
    
    Returns weather data including METAR, TAF, PIREPs, and SIGMETs for each waypoint
    """
    weather_service = WeatherService()
    
    try:
        # Parse the route into waypoints
        waypoints = weather_service.parse_route(route)
        
        # Fetch and process weather data for each waypoint
        results = []
        for waypoint in waypoints:
            icao = waypoint["icao"]
            altitude = waypoint["altitude"]
            
            # Fetch weather products
            raw_metar = weather_service.fetch_metar(icao)
            raw_taf = weather_service.fetch_taf(icao)
            raw_pireps = weather_service.fetch_pireps(icao)
            raw_sigmets = weather_service.fetch_sigmets(icao)
            
            # Decode weather products
            decoded_metar = weather_service.decode_metar(raw_metar)
            decoded_taf = weather_service.decode_taf(raw_taf)
            decoded_pireps = weather_service.decode_pireps(raw_pireps)
            decoded_sigmets = weather_service.decode_sigmets(raw_sigmets)
            
            # Filter PIREPs by altitude relevance (within 5000 feet of waypoint altitude)
            relevant_pireps = []
            for pirep in decoded_pireps:
                pirep_altitude = pirep.get("altitude")
                if pirep_altitude is not None:
                    if abs(pirep_altitude - altitude) <= 5000:
                        relevant_pireps.append(pirep)
                else:
                    # Include PIREPs without altitude information
                    relevant_pireps.append(pirep)
            
            # Classify weather based on all available data
            classification = weather_service.classify_weather(
                decoded_metar,
                relevant_pireps,
                decoded_sigmets
            )
            
            # Add to results
            results.append({
                "icao": icao,
                "altitude": altitude,
                "metar": {
                    "raw": raw_metar,
                    "summary": decoded_metar["summary"]
                },
                "taf": {
                    "raw": raw_taf,
                    "summary": decoded_taf["summary"]
                },
                "pireps": [
                    {
                        "raw": pirep["raw"],
                        "summary": pirep["summary"],
                        "altitude": pirep.get("altitude")
                    } for pirep in relevant_pireps
                ],
                "sigmets": [
                    {
                        "raw": sigmet["raw"],
                        "summary": sigmet["summary"]
                    } for sigmet in decoded_sigmets
                ],
                "classification": classification
            })
        
        return results
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Catch any other exceptions
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)