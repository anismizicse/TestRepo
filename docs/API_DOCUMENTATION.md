# API Documentation - Quarkus Users API

Complete reference for all REST API endpoints, their parameters, responses, and usage examples.

---

## 📋 Table of Contents
1. [API Overview](#api-overview)
2. [Base Configuration](#base-configuration)
3. [HTTP Methods](#http-methods)
4. [Response Codes](#response-codes)
5. [User Data Model](#user-data-model)
6. [Authentication & Security](#authentication--security)
7. [CRUD Endpoints](#crud-endpoints)
8. [Search Endpoints](#search-endpoints)
9. [Error Handling](#error-handling)
10. [Example Workflows](#example-workflows)
11. [Testing with Postman](#testing-with-postman)
12. [Request/Response Examples](#requestresponse-examples)

---

## 📡 API Overview

This is a **RESTful API** for managing user data stored in MongoDB.

**Base URL:** `http://localhost:8080`  
**API Prefix:** `/api/users`  
**Data Format:** JSON  
**Content-Type:** `application/json`

---

## ⚙️ Base Configuration

### Server Running Locally

```
Protocol: HTTP
Host: localhost
Port: 8080
Base Path: /api/users

Full URL Pattern: http://localhost:8080/api/users[/path][?params]
```

### Production Environment

When deployed to cloud:
```
Protocol: HTTPS
Host: your-api-domain.com
Port: 443 (default for HTTPS)
Base Path: /api/users

Full URL Pattern: https://your-api-domain.com/api/users[/path][?params]
```

### Required Headers

All requests must include:
```http
Content-Type: application/json
```

---

## 🔄 HTTP Methods

| Method | Purpose | Example |
|--------|---------|---------|
| **GET** | Retrieve data | Get all users, Get single user |
| **POST** | Create new data | Create new user |
| **PUT** | Update existing data | Update user information |
| **DELETE** | Remove data | Delete user |

### Method Details

**GET** - Safe & Idempotent
- Doesn't modify data
- Multiple calls return same result
- Browser can cache response

**POST** - Creates New Resource
- Returns 201 Created with new resource
- Each call creates new data
- Not idempotent (multiple calls create multiple resources)

**PUT** - Replaces Resource
- Replaces entire resource
- Returns updated resource
- Idempotent (multiple identical calls have same effect)

**DELETE** - Removes Resource
- Returns 204 No Content on success
- Returns 404 if resource doesn't exist
- Idempotent (multiple deletes of same resource OK)

---

## ✅ Response Status Codes

| Code | Status | Meaning | Example |
|------|--------|---------|---------|
| **200** | OK | Request succeeded | GET user returned successfully |
| **201** | Created | Resource created | User created successfully |
| **204** | No Content | Success, no response body | User deleted successfully |
| **400** | Bad Request | Invalid request data | Missing required field |
| **404** | Not Found | Resource doesn't exist | User ID doesn't exist |
| **500** | Server Error | Server error | Database connection failed |

### Response Status Details

**200 OK**
- Used for: GET, PUT operations that return data
- Response body: Data returned (user object or array)

**201 Created**
- Used for: POST operations
- Response body: Newly created resource (user object with new ID)
- Header: `Location: /api/users/{id}`

**204 No Content**
- Used for: DELETE operations
- Response body: Empty (no content)
- Indicates successful deletion

**400 Bad Request**
- Used for: Invalid input data
- Response body: Error message
- Common causes:
  - Missing required fields
  - Invalid data types
  - Invalid ID format

**404 Not Found**
- Used for: Accessing non-existent resource
- Response body: Error message
- Common causes:
  - User ID doesn't exist
  - Wrong endpoint path

**500 Server Error**
- Used for: Internal server errors
- Response body: Error message
- Causes: Database down, unexpected exception

---

## 👤 User Data Model

### User Object Structure

```json
{
  "id": "507f1f77bcf86cd799439011",
  "firstName": "John",
  "lastName": "Doe",
  "email": "john.doe@example.com",
  "phoneNumber": "+1-555-0123",
  "city": "New York",
  "age": 28
}
```

### Field Specifications

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `id` | String | Auto | MongoDB ObjectId (auto-generated) | `507f1f77bcf86cd799439011` |
| `firstName` | String | Yes | User's first name | `John` |
| `lastName` | String | Yes | User's last name | `Doe` |
| `email` | String | Yes | Email address | `john.doe@example.com` |
| `phoneNumber` | String | Yes | Phone number with format | `+1-555-0123` |
| `city` | String | Yes | City of residence | `New York` |
| `age` | Integer | Yes | User's age | `28` |

### Field Validation

- **firstName**: Required, non-empty string
- **lastName**: Required, non-empty string
- **email**: Required, must be valid email format (contains @)
- **phoneNumber**: Required, recommend E.164 format (+1-555-0000)
- **city**: Required, non-empty string
- **age**: Required, must be positive integer (0-150 recommended)

### Field Constraints

```
firstName:    1-50 characters
lastName:     1-50 characters
email:        5-100 characters, must contain @
phoneNumber:  8-20 characters, can include +, -, (), spaces
city:         1-50 characters
age:          0-150 (reasonable limits)
```

---

## 🔐 Authentication & Security

### Current Implementation
- **No authentication required** (development mode)
- All endpoints publicly accessible
- No API keys or tokens needed

### Production Recommendations
1. **Add JWT Authentication**
   - Token-based authentication
   - Include token in Authorization header
   
2. **Use HTTPS**
   - Encrypt data in transit
   - Prevent man-in-the-middle attacks

3. **Add Rate Limiting**
   - Limit requests per IP/user
   - Prevent abuse and brute force

4. **Add Input Validation**
   - Validate all input data
   - Sanitize before storing

5. **Add CORS Policy**
   - Control which origins can access API
   - Prevent unauthorized cross-origin requests

### Example Production Header
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

---

## 🎯 CRUD Endpoints

### CREATE - Add New User

**Endpoint:** `POST /api/users`

**Description:** Create a new user in the database

**Request Headers:**
```http
Content-Type: application/json
```

**Request Body:**
```json
{
  "firstName": "John",
  "lastName": "Doe",
  "email": "john.doe@example.com",
  "phoneNumber": "+1-555-0123",
  "city": "New York",
  "age": 28
}
```

**Response Code:** `201 Created`

**Response Body:**
```json
{
  "id": "507f1f77bcf86cd799439011",
  "firstName": "John",
  "lastName": "Doe",
  "email": "john.doe@example.com",
  "phoneNumber": "+1-555-0123",
  "city": "New York",
  "age": 28
}
```

**Notes:**
- `id` field is auto-generated, do NOT include in request
- All fields except `id` are required
- Returns full user object with new ID
- Status code 201 indicates successful creation

**curl Example:**
```bash
curl -X POST http://localhost:8080/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "firstName": "John",
    "lastName": "Doe",
    "email": "john.doe@example.com",
    "phoneNumber": "+1-555-0123",
    "city": "New York",
    "age": 28
  }'
```

---

### READ - Get All Users

**Endpoint:** `GET /api/users`

**Description:** Retrieve all users from database

**Request Headers:**
```http
Accept: application/json
```

**Request Parameters:** None

**Response Code:** `200 OK`

**Response Body:**
```json
[
  {
    "id": "507f1f77bcf86cd799439011",
    "firstName": "John",
    "lastName": "Doe",
    "email": "john.doe@example.com",
    "phoneNumber": "+1-555-0123",
    "city": "New York",
    "age": 28
  },
  {
    "id": "507f1f77bcf86cd799439012",
    "firstName": "Jane",
    "lastName": "Smith",
    "email": "jane.smith@example.com",
    "phoneNumber": "+1-555-0456",
    "city": "Boston",
    "age": 26
  }
]
```

**Notes:**
- Returns array of user objects
- Empty array `[]` if no users exist
- Always returns 200 OK, even if empty
- Returns all users (consider pagination for large datasets)

**curl Example:**
```bash
curl http://localhost:8080/api/users
```

---

### READ - Get Single User

**Endpoint:** `GET /api/users/{id}`

**Description:** Retrieve specific user by ID

**Request Headers:**
```http
Accept: application/json
```

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | String | Yes | MongoDB ObjectId of user |

**Response Code:** `200 OK` or `404 Not Found`

**Response Body (200 OK):**
```json
{
  "id": "507f1f77bcf86cd799439011",
  "firstName": "John",
  "lastName": "Doe",
  "email": "john.doe@example.com",
  "phoneNumber": "+1-555-0123",
  "city": "New York",
  "age": 28
}
```

**Response Body (404 Not Found):**
```json
"User not found with id: 507f1f77bcf86cd799439099"
```

**Notes:**
- ID must be valid MongoDB ObjectId format
- Returns single user object
- Returns 404 if ID not found
- Invalid ID format returns 400 Bad Request

**curl Example:**
```bash
curl http://localhost:8080/api/users/507f1f77bcf86cd799439011
```

---

### UPDATE - Modify User

**Endpoint:** `PUT /api/users/{id}`

**Description:** Update existing user information

**Request Headers:**
```http
Content-Type: application/json
```

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | String | Yes | MongoDB ObjectId of user |

**Request Body:**
```json
{
  "firstName": "John",
  "lastName": "Williams",
  "email": "john.williams@example.com",
  "phoneNumber": "+1-555-9999",
  "city": "San Francisco",
  "age": 29
}
```

**Response Code:** `200 OK` or `404 Not Found`

**Response Body (200 OK):**
```json
{
  "id": "507f1f77bcf86cd799439011",
  "firstName": "John",
  "lastName": "Williams",
  "email": "john.williams@example.com",
  "phoneNumber": "+1-555-9999",
  "city": "San Francisco",
  "age": 29
}
```

**Response Body (404 Not Found):**
```json
"User not found with id: 507f1f77bcf86cd799439099"
```

**Notes:**
- All fields must be provided in request body
- Cannot update `id` field
- Returns updated user object
- Returns 404 if user doesn't exist
- Status 200 indicates successful update

**curl Example:**
```bash
curl -X PUT http://localhost:8080/api/users/507f1f77bcf86cd799439011 \
  -H "Content-Type: application/json" \
  -d '{
    "firstName": "John",
    "lastName": "Williams",
    "email": "john.williams@example.com",
    "phoneNumber": "+1-555-9999",
    "city": "San Francisco",
    "age": 29
  }'
```

---

### DELETE - Remove User

**Endpoint:** `DELETE /api/users/{id}`

**Description:** Delete user from database

**Request Headers:** None

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | String | Yes | MongoDB ObjectId of user |

**Response Code:** `204 No Content` or `404 Not Found`

**Response Body (204 No Content):**
```
(empty)
```

**Response Body (404 Not Found):**
```json
"User not found with id: 507f1f77bcf86cd799439099"
```

**Notes:**
- No response body for successful delete (204)
- Returns 404 if user doesn't exist
- Delete is permanent - cannot undo
- Idempotent - multiple deletes of same ID have same effect

**curl Example:**
```bash
curl -X DELETE http://localhost:8080/api/users/507f1f77bcf86cd799439011
```

---

## 🔍 Search Endpoints

### Search by City

**Endpoint:** `GET /api/users/search/city`

**Description:** Find all users in a specific city

**Request Headers:**
```http
Accept: application/json
```

**Request Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `city` | String | Yes | City name to search | `New York` |

**Response Code:** `200 OK`

**Response Body:**
```json
[
  {
    "id": "507f1f77bcf86cd799439011",
    "firstName": "John",
    "lastName": "Doe",
    "email": "john.doe@example.com",
    "phoneNumber": "+1-555-0123",
    "city": "New York",
    "age": 28
  },
  {
    "id": "507f1f77bcf86cd799439013",
    "firstName": "Bob",
    "lastName": "Johnson",
    "email": "bob.johnson@example.com",
    "phoneNumber": "+1-555-0789",
    "city": "New York",
    "age": 32
  }
]
```

**Notes:**
- Case-sensitive search
- Returns array of matching users
- Empty array `[]` if no matches
- City parameter is required
- URL encode spaces as %20

**curl Examples:**
```bash
# Search for New York users
curl "http://localhost:8080/api/users/search/city?city=New%20York"

# Search for Los Angeles users
curl "http://localhost:8080/api/users/search/city?city=Los%20Angeles"

# Search for Boston users
curl "http://localhost:8080/api/users/search/city?city=Boston"
```

---

### Search by Age Range

**Endpoint:** `GET /api/users/search/age`

**Description:** Find users within age range

**Request Headers:**
```http
Accept: application/json
```

**Request Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `minAge` | Integer | Yes | Minimum age (inclusive) | `20` |
| `maxAge` | Integer | Yes | Maximum age (inclusive) | `30` |

**Response Code:** `200 OK`

**Response Body:**
```json
[
  {
    "id": "507f1f77bcf86cd799439011",
    "firstName": "John",
    "lastName": "Doe",
    "email": "john.doe@example.com",
    "phoneNumber": "+1-555-0123",
    "city": "New York",
    "age": 28
  },
  {
    "id": "507f1f77bcf86cd799439012",
    "firstName": "Jane",
    "lastName": "Smith",
    "email": "jane.smith@example.com",
    "phoneNumber": "+1-555-0456",
    "city": "Boston",
    "age": 26
  }
]
```

**Notes:**
- Both minAge and maxAge required
- Inclusive range (minAge ≤ age ≤ maxAge)
- Returns array of matching users
- Empty array if no matches
- Returns users whose age is within specified range

**curl Examples:**
```bash
# Users aged 20-30
curl "http://localhost:8080/api/users/search/age?minAge=20&maxAge=30"

# Users aged 25-35
curl "http://localhost:8080/api/users/search/age?minAge=25&maxAge=35"

# Users aged 18-25
curl "http://localhost:8080/api/users/search/age?minAge=18&maxAge=25"
```

---

## ⚠️ Error Handling

### Error Response Format

All errors return JSON with descriptive message:

```json
"Error message describing what went wrong"
```

### Common Error Scenarios

**1. Missing Required Fields**

**Request:**
```json
{
  "firstName": "John"
}
```

**Response:** `400 Bad Request`
```json
"Missing required field: lastName"
```

**Fix:** Include all required fields in request

---

**2. Invalid ID Format**

**Request:** `GET /api/users/invalid-id-format`

**Response:** `400 Bad Request`
```json
"Invalid user ID format"
```

**Fix:** Use valid MongoDB ObjectId format (24 hex characters)

---

**3. User Not Found**

**Request:** `GET /api/users/507f1f77bcf86cd799439099`

**Response:** `404 Not Found`
```json
"User not found with id: 507f1f77bcf86cd799439099"
```

**Fix:** Verify user exists, check ID spelling

---

**4. Invalid Content-Type**

**Request:** Missing `Content-Type: application/json` header

**Response:** `400 Bad Request`
```json
"Invalid content type, expected application/json"
```

**Fix:** Add `Content-Type: application/json` header

---

**5. Database Connection Error**

**Response:** `500 Server Error`
```json
"Database connection failed"
```

**Fix:** Verify MongoDB is running

---

### Error Handling Best Practices

1. **Always check status code** before processing response
2. **Handle 4xx errors** - Fix request (user error)
3. **Handle 5xx errors** - Retry or contact support (server error)
4. **Read error message** - Understand what went wrong
5. **Log errors** - For debugging and monitoring

### Example Error Handling (JavaScript)

```javascript
fetch('http://localhost:8080/api/users', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({...userData})
})
.then(response => {
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
})
.then(data => console.log('Success:', data))
.catch(error => console.error('Error:', error));
```

---

## 📚 Example Workflows

### Workflow 1: Create User and Get ID

**Step 1:** Create user
```bash
curl -X POST http://localhost:8080/api/users \
  -H "Content-Type: application/json" \
  -d '{"firstName":"John","lastName":"Doe",...}'
```

**Response:**
```json
{
  "id": "507f1f77bcf86cd799439011",
  "firstName": "John",
  ...
}
```

**Step 2:** Store ID from response
```
id = "507f1f77bcf86cd799439011"
```

**Step 3:** Use ID for subsequent calls
```bash
curl http://localhost:8080/api/users/507f1f77bcf86cd799439011
```

---

### Workflow 2: Update User Information

**Step 1:** Get current user
```bash
curl http://localhost:8080/api/users/507f1f77bcf86cd799439011
```

**Step 2:** Modify data
```json
{
  "firstName": "John",
  "lastName": "Williams",    // Changed
  "email": "new@example.com", // Changed
  "phoneNumber": "+1-555-9999",
  "city": "San Francisco",    // Changed
  "age": 29                   // Changed
}
```

**Step 3:** Send updated data
```bash
curl -X PUT http://localhost:8080/api/users/507f1f77bcf86cd799439011 \
  -H "Content-Type: application/json" \
  -d '{...updated data...}'
```

---

### Workflow 3: Search and Filter

**Step 1:** Find users in specific city
```bash
curl "http://localhost:8080/api/users/search/city?city=New%20York"
```

**Response:** Array of New York users

**Step 2:** Find users in age range
```bash
curl "http://localhost:8080/api/users/search/age?minAge=25&maxAge=35"
```

**Response:** Array of users aged 25-35

---

### Workflow 4: Delete User

**Step 1:** Get all users
```bash
curl http://localhost:8080/api/users
```

**Step 2:** Find user to delete
```
Look for user with id: "507f1f77bcf86cd799439011"
```

**Step 3:** Delete user
```bash
curl -X DELETE http://localhost:8080/api/users/507f1f77bcf86cd799439011
```

**Response:** 204 No Content (no body)

**Step 4:** Verify deletion
```bash
curl http://localhost:8080/api/users/507f1f77bcf86cd799439011
```

**Response:** 404 Not Found (user no longer exists)

---

## 🧪 Testing with Postman

### Import Collection

1. Open Postman
2. Click **File** → **Import**
3. Select `Quarkus_Users_API.postman_collection.json`
4. All endpoints appear in left sidebar

### Test Each Endpoint

**Collection contains 10 pre-configured requests:**

1. **Create User - Alice** (POST)
   - Click Send
   - Verify 201 status
   - Copy `id` from response

2. **Get All Users** (GET)
   - Shows all users
   - Verify Alice appears

3. **Get User by ID** (GET)
   - Click Send
   - Use ID from previous step

4. **Update User** (PUT)
   - Modify data in request body
   - Click Send
   - Verify updated data in response

5. **Delete User** (DELETE)
   - Click Send
   - Verify 204 No Content

6-10. **Additional test requests**
   - Search by city
   - Search by age
   - Create multiple users

---

## 📤 Request/Response Examples

### Complete Example: Create and Search

**Step 1: Create User**

```bash
curl -X POST http://localhost:8080/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "firstName": "Alice",
    "lastName": "Johnson",
    "email": "alice@example.com",
    "phoneNumber": "+1-555-7890",
    "city": "Los Angeles",
    "age": 25
  }'
```

**Response:** `201 Created`
```json
{
  "id": "507f1f77bcf86cd799439011",
  "firstName": "Alice",
  "lastName": "Johnson",
  "email": "alice@example.com",
  "phoneNumber": "+1-555-7890",
  "city": "Los Angeles",
  "age": 25
}
```

---

**Step 2: Search by City**

```bash
curl "http://localhost:8080/api/users/search/city?city=Los%20Angeles"
```

**Response:** `200 OK`
```json
[
  {
    "id": "507f1f77bcf86cd799439011",
    "firstName": "Alice",
    "lastName": "Johnson",
    "email": "alice@example.com",
    "phoneNumber": "+1-555-7890",
    "city": "Los Angeles",
    "age": 25
  }
]
```

---

**Step 3: Update User**

```bash
curl -X PUT http://localhost:8080/api/users/507f1f77bcf86cd799439011 \
  -H "Content-Type: application/json" \
  -d '{
    "firstName": "Alice",
    "lastName": "Williams",
    "email": "alice.w@example.com",
    "phoneNumber": "+1-555-9999",
    "city": "San Francisco",
    "age": 26
  }'
```

**Response:** `200 OK`
```json
{
  "id": "507f1f77bcf86cd799439011",
  "firstName": "Alice",
  "lastName": "Williams",
  "email": "alice.w@example.com",
  "phoneNumber": "+1-555-9999",
  "city": "San Francisco",
  "age": 26
}
```

---

**Step 4: Delete User**

```bash
curl -X DELETE http://localhost:8080/api/users/507f1f77bcf86cd799439011
```

**Response:** `204 No Content`
```
(empty body)
```

---

## 🚀 API Testing Checklist

Before deploying, verify:

```
Endpoint Testing:
☐ POST /api/users - Create works, returns 201
☐ GET /api/users - Get all works, returns array
☐ GET /api/users/{id} - Get single works, returns object
☐ PUT /api/users/{id} - Update works, returns updated object
☐ DELETE /api/users/{id} - Delete works, returns 204
☐ GET /api/users/search/city?city=X - Search city works
☐ GET /api/users/search/age?minAge=X&maxAge=Y - Search age works

Error Cases:
☐ Invalid ID format returns 400
☐ Non-existent ID returns 404
☐ Missing fields returns 400
☐ Invalid method returns 405
☐ Database down returns 500

Data Validation:
☐ All fields required
☐ Email format validated
☐ Age is positive integer
☐ No duplicate IDs generated
☐ Timestamps working
```

---

## 📖 Summary

| Operation | Method | Endpoint | Status | Usage |
|-----------|--------|----------|--------|-------|
| Create | POST | `/api/users` | 201 | Create new user |
| Read All | GET | `/api/users` | 200 | Get all users |
| Read One | GET | `/api/users/{id}` | 200 | Get single user |
| Update | PUT | `/api/users/{id}` | 200 | Update user |
| Delete | DELETE | `/api/users/{id}` | 204 | Delete user |
| Search City | GET | `/api/users/search/city?city=X` | 200 | Find by city |
| Search Age | GET | `/api/users/search/age?minAge=X&maxAge=Y` | 200 | Find by age |

---

**Last Updated:** November 5, 2025  
**API Version:** 1.0.0  
**Base URL:** http://localhost:8080/api/users  
**Data Format:** JSON
