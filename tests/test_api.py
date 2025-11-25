"""
Test script for Marvel Knowledge Graph API (Phase 5).

Tests all API endpoints with sample data to ensure proper functionality.
"""

import sys
import time
import requests
import json
from pathlib import Path
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_CHARACTERS = [
    "Spider-Man (Peter Parker)",
    "Captain America (Steven Rogers)",
    "Thor (Thor Odinson)"
]


# ============================================================================
# Test Utilities
# ============================================================================

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def print_test(test_name: str):
    """Print a test name."""
    print(f"\n{'─' * 80}")
    print(f"🧪 {test_name}")
    print(f"{'─' * 80}")


def print_success(message: str):
    """Print success message."""
    print(f"✅ {message}")


def print_error(message: str):
    """Print error message."""
    print(f"❌ {message}")


def print_info(message: str):
    """Print info message."""
    print(f"ℹ️  {message}")


def print_json(data: Any, max_length: int = 500):
    """Print JSON data (truncated if too long)."""
    json_str = json.dumps(data, indent=2)
    if len(json_str) > max_length:
        print(json_str[:max_length] + "\n... (truncated)")
    else:
        print(json_str)


# ============================================================================
# Test Functions
# ============================================================================

def test_health_check() -> bool:
    """Test the /health endpoint."""
    print_test("Health Check")

    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print_json(data)

            if data.get("status") == "healthy":
                print_success("API is healthy")
                return True
            else:
                print_error(f"API status: {data.get('status')}")
                return False
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


def test_root_endpoint() -> bool:
    """Test the root / endpoint."""
    print_test("Root Endpoint")

    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print_json(data)
            print_success("Root endpoint working")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Root endpoint failed: {e}")
        return False


def test_question_endpoint() -> bool:
    """Test the POST /question endpoint."""
    print_test("Natural Language Questions")

    questions = [
        "How did Spider-Man get his powers?",
        "What are Thor's abilities?",
        "Why do Captain America's powers matter?",
    ]

    success_count = 0

    for question in questions:
        print(f"\n📝 Question: \"{question}\"")

        try:
            response = requests.post(
                f"{API_BASE_URL}/question",
                json={
                    "question": question,
                    "verbose": False,
                    "include_context": False
                }
            )

            print(f"   Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"   Answer: {data.get('answer', 'N/A')[:200]}...")
                print(f"   Query Type: {data.get('query_type', 'N/A')}")
                print(f"   Characters: {data.get('characters', [])}")
                print(f"   Confidence: {data.get('confidence_level', 'N/A')}")
                print_success("Question answered")
                success_count += 1
            else:
                print_error(f"Failed with status {response.status_code}")
                print(f"   Response: {response.text}")

        except Exception as e:
            print_error(f"Question failed: {e}")

        time.sleep(0.5)  # Rate limiting

    print(f"\n✅ Successfully answered {success_count}/{len(questions)} questions")
    return success_count == len(questions)


def test_graph_endpoint() -> bool:
    """Test the GET /graph/{character} endpoint."""
    print_test("Character Graph View")

    success_count = 0

    for char_name in TEST_CHARACTERS:
        print(f"\n👤 Character: {char_name}")

        try:
            response = requests.get(
                f"{API_BASE_URL}/graph/{char_name}",
                params={"search_by": "name"}
            )

            print(f"   Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                character = data.get("character", {})
                origin = data.get("power_origin", {})
                powers = data.get("powers", [])

                print(f"   Name: {character.get('name', 'N/A')}")
                print(f"   ID: {character.get('node_id', 'N/A')}")
                print(f"   Alignment: {character.get('alignment', 'N/A')}")
                print(f"   Origin Type: {origin.get('origin_type', 'N/A')}")
                print(f"   Confidence: {origin.get('confidence', 'N/A')}")
                print(f"   Powers: {len(powers)} abilities")

                print_success("Graph data retrieved")
                success_count += 1
            elif response.status_code == 404:
                print_error(f"Character not found: {char_name}")
            else:
                print_error(f"Failed with status {response.status_code}")

        except Exception as e:
            print_error(f"Graph query failed: {e}")

        time.sleep(0.3)

    print(f"\n✅ Successfully retrieved {success_count}/{len(TEST_CHARACTERS)} character graphs")
    return success_count > 0


def test_extraction_report_endpoint() -> bool:
    """Test the GET /extraction-report/{character} endpoint."""
    print_test("Extraction Reports")

    success_count = 0

    for char_name in TEST_CHARACTERS:
        print(f"\n📊 Report for: {char_name}")

        try:
            response = requests.get(
                f"{API_BASE_URL}/extraction-report/{char_name}",
                params={
                    "search_by": "name",
                    "include_extraction_data": False
                }
            )

            print(f"   Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                print(f"   Validation Passed: {data.get('validation_passed', False)}")
                print(f"   Confidence Score: {data.get('confidence_score', 0):.3f}")
                print(f"   Completeness Score: {data.get('completeness_score', 0):.3f}")
                print(f"   Semantic Similarity: {data.get('semantic_similarity', 0):.3f}")
                print(f"   Quality Tier: {data.get('quality_tier', 'N/A')}")
                print(f"   Strengths: {len(data.get('strengths', []))}")
                print(f"   Weaknesses: {len(data.get('weaknesses', []))}")
                print(f"   Recommendations: {len(data.get('recommendations', []))}")

                print_success("Extraction report retrieved")
                success_count += 1
            elif response.status_code == 404:
                print_error(f"Report not found for: {char_name}")
            else:
                print_error(f"Failed with status {response.status_code}")

        except Exception as e:
            print_error(f"Report query failed: {e}")

        time.sleep(0.3)

    print(f"\n✅ Successfully retrieved {success_count}/{len(TEST_CHARACTERS)} extraction reports")
    return success_count > 0


def test_validate_extraction_endpoint() -> bool:
    """Test the POST /validate-extraction endpoint."""
    print_test("Re-validation")

    print(f"\n🔍 Validating: {TEST_CHARACTERS[0]}")

    try:
        response = requests.post(
            f"{API_BASE_URL}/validate-extraction",
            json={
                "character_name": TEST_CHARACTERS[0],
                "enable_multi_pass": False,
                "verbose": False
            }
        )

        print(f"   Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print(f"   Character: {data.get('character_name', 'N/A')}")
            print(f"   Validation Passed: {data.get('validation_passed', False)}")
            print(f"   Confidence: {data.get('confidence_score', 0):.3f}")
            print(f"   Completeness: {data.get('completeness_score', 0):.3f}")
            print(f"   Message: {data.get('message', 'N/A')}")

            print_success("Validation completed")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print_error(f"Validation failed: {e}")
        return False


def test_list_characters_endpoint() -> bool:
    """Test the GET /characters endpoint."""
    print_test("List Characters")

    try:
        response = requests.get(
            f"{API_BASE_URL}/characters",
            params={
                "limit": 10,
                "offset": 0
            }
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print(f"   Total Characters: {data.get('total', 0)}")
            print(f"   Returned: {data.get('count', 0)}")
            print(f"   Limit: {data.get('limit', 0)}")
            print(f"   Offset: {data.get('offset', 0)}")

            characters = data.get("characters", [])
            if characters:
                print(f"\n   Sample Characters:")
                for char in characters[:5]:
                    print(f"   - {char.get('name', 'N/A')} ({char.get('alignment', 'N/A')})")

            print_success("Character list retrieved")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False

    except Exception as e:
        print_error(f"List characters failed: {e}")
        return False


def test_stats_endpoint() -> bool:
    """Test the GET /stats endpoint."""
    print_test("Graph Statistics")

    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print_json(data, max_length=1000)
            print_success("Graph statistics retrieved")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Stats query failed: {e}")
        return False


def test_error_handling() -> bool:
    """Test error handling."""
    print_test("Error Handling")

    tests_passed = 0
    total_tests = 2

    # Test 1: Character not found
    print("\n🔍 Test: Character not found (404)")
    try:
        response = requests.get(f"{API_BASE_URL}/graph/NonexistentCharacter12345")
        print(f"   Status Code: {response.status_code}")

        if response.status_code == 404:
            print_success("404 error handled correctly")
            tests_passed += 1
        else:
            print_error(f"Expected 404, got {response.status_code}")

    except Exception as e:
        print_error(f"Error test failed: {e}")

    # Test 2: Invalid endpoint
    print("\n🔍 Test: Invalid endpoint (404)")
    try:
        response = requests.get(f"{API_BASE_URL}/invalid-endpoint-xyz")
        print(f"   Status Code: {response.status_code}")

        if response.status_code == 404:
            print_success("404 error handled correctly")
            tests_passed += 1
        else:
            print_error(f"Expected 404, got {response.status_code}")

    except Exception as e:
        print_error(f"Error test failed: {e}")

    print(f"\n✅ Passed {tests_passed}/{total_tests} error handling tests")
    return tests_passed == total_tests


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all API tests."""
    print_section("PHASE 5: API & Integration Test Suite")

    print_info("Testing Marvel Knowledge Graph API")
    print_info(f"Base URL: {API_BASE_URL}")
    print_info("Ensure the API server is running before running tests!")
    print_info("Run: python -m src.api.main")

    # Check if API is reachable
    print("\n🔌 Checking API connectivity...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print_success("API is reachable")
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Make sure the server is running!")
        print_info("Start server with: python -m src.api.main")
        return
    except Exception as e:
        print_error(f"Connection error: {e}")
        return

    # Run tests
    results = {
        "Health Check": test_health_check(),
        "Root Endpoint": test_root_endpoint(),
        "Natural Language Questions": test_question_endpoint(),
        "Character Graph View": test_graph_endpoint(),
        "Extraction Reports": test_extraction_report_endpoint(),
        "Re-validation": test_validate_extraction_endpoint(),
        "List Characters": test_list_characters_endpoint(),
        "Graph Statistics": test_stats_endpoint(),
        "Error Handling": test_error_handling(),
    }

    # Summary
    print_section("TEST SUMMARY")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"\nResults by Test Category:\n")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\n{'=' * 80}")
    print(f"Overall: {passed}/{total} test categories passed")
    print(f"{'=' * 80}\n")

    if passed == total:
        print("🎉 All tests passed! Phase 5 is complete!")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please review the output above.")

    print("\n📚 API Documentation: http://localhost:8000/docs")
    print("🔍 ReDoc: http://localhost:8000/redoc")
    print("❤️  Health: http://localhost:8000/health\n")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    run_all_tests()
