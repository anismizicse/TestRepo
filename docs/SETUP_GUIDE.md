# Complete Setup Guide - Quarkus MongoDB CRUD API

Complete guide covering all setup, installation, and configuration steps needed to get this Quarkus API running.

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [MongoDB Installation & Setup](#mongodb-installation--setup)
3. [Project Setup](#project-setup)
4. [Running the Application](#running-the-application)
5. [Postman Setup & Testing](#postman-setup--testing)
6. [Verify Everything Works](#verify-everything-works)
7. [Troubleshooting](#troubleshooting)
8. [MongoDB Compass (Optional)](#mongodb-compass-optional)
9. [Deployment](#deployment)

---

## 📦 Prerequisites

### System Requirements
- **macOS** (or Linux/Windows with adjustments)
- **Java 17 or higher** installed and in PATH
- **Maven 3.8.1 or higher** (included with project via mvnw)
- **Internet connection** for downloading dependencies

### Verify Prerequisites
```bash
# Check Java version
java -version
# Output should show: openjdk version "17.X.X" or higher

# Check Maven
./mvnw -v
# Output should show: Apache Maven 3.8.1 or higher
```

If these commands fail, proceed to the MongoDB setup and they will be verified during first build.

---

## 🗄️ MongoDB Installation & Setup

MongoDB is the database that stores user data. Choose one installation method:

### Method 1: Homebrew (macOS) - Recommended ✅

**Step 1: Add MongoDB Tap**
```bash
brew tap mongodb/brew
```

**Step 2: Install MongoDB Community Edition**
```bash
brew install mongodb-community
```

**Step 3: Start MongoDB Service**
```bash
# Start MongoDB as a service (runs in background)
brew services start mongodb-community

# Output: Successfully started `mongodb-community`
```

**Step 4: Verify MongoDB is Running**
```bash
# Open MongoDB shell
mongosh

# You should see: "test>"

# Check version
db.version()

# You should see: "8.2.1" (or similar)

# Exit MongoDB shell
exit
```

**To Stop MongoDB Later:**
```bash
brew services stop mongodb-community
```

---

### Method 2: Docker

**Prerequisites:** Docker Desktop installed

**Step 1: Create MongoDB Container**
```bash
docker run -d \
  --name quarkus-mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:8.2.1
```

**Step 2: Verify Container is Running**
```bash
docker ps | grep quarkus-mongodb
```

**Step 3: Connect to MongoDB**
```bash
mongosh "mongodb://admin:password@localhost:27017"
```

**To Stop Container Later:**
```bash
docker stop quarkus-mongodb
docker rm quarkus-mongodb
```

---

### Method 3: Manual Installation (macOS/Linux)

1. Download from: https://www.mongodb.com/try/download/community
2. Extract to `/usr/local/mongodb`
3. Create data directory: `mkdir -p ~/data/db`
4. Start server: `/usr/local/mongodb/bin/mongod --dbpath ~/data/db`
5. Verify: `/usr/local/mongodb/bin/mongosh`

---

## 🚀 Project Setup

### Step 1: Navigate to Project Directory
```bash
cd /Users/bjit/Desktop/My_Files/Projects/Quarkus/getting-started
```

### Step 2: Build the Project (First Time Only)
```bash
./mvnw clean compile
```

**Expected Output:**
```
[INFO] -------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] -------------------------------------------------------
[INFO] Total time: XX.XXX s
```

This downloads dependencies (first run takes ~2-3 minutes).

### Step 3: Verify Source Files Exist
```bash
# Check if all Java files are present
ls -la src/main/java/org/acme/

# Should show:
# User.java
# UserRepository.java
# UserService.java
# UserResource.java
```

### Step 4: Verify Configuration File
```bash
# Check application.properties
cat src/main/resources/application.properties

# Should show MongoDB connection details:
# quarkus.mongodb.connection-string=mongodb://localhost:27017
# quarkus.mongodb.database=quarkus_users
```

---

## ▶️ Running the Application

### Start Quarkus in Development Mode

```bash
./mvnw quarkus:dev
```

**Expected Output:**
```
[INFO] Quarkus 3.29.0 on JVM started in 4.235s.
[INFO] Profile prod activated.
[INFO] Installed features: cdi, jackson, mongodb-panache-reactive, resteasy-reactive, etc.
[INFO] Listening on: http://localhost:8080
[INFO] Application metrics available at: http://localhost:9000/q/metrics
```

**Application is ready when you see: "Listening on: http://localhost:8080"**

---

### Verify Application is Running

**In a new terminal:**
```bash
# Test the API is responding
curl http://localhost:8080/api/users

# Should return:
# []
# (empty array means no users yet, which is correct)
```

---

## 📮 Postman Setup & Testing

Postman is a tool for testing API endpoints without writing code.

### Step 1: Install Postman

**Download from:** https://www.postman.com/downloads/

Choose your operating system and install.

### Step 2: Import Collection

**Option A: From File (Recommended)**

1. Open Postman
2. Click **File** → **Import**
3. Select: `Quarkus_Users_API.postman_collection.json`
4. Click **Import**
5. You should see "Quarkus Users API" in left sidebar with 10 requests

**Option B: Manual Setup**

If file import doesn't work, create requests manually:

1. **Create User (POST)**
   - Method: `POST`
   - URL: `http://localhost:8080/api/users`
   - Headers: `Content-Type: application/json`
   - Body:
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
   - Click **Send**
   - Should return `201 Created` with user data + `_id` field

2. **Get All Users (GET)**
   - Method: `GET`
   - URL: `http://localhost:8080/api/users`
   - Click **Send**
   - Should return array of all users

---

### Step 3: Test All API Endpoints

#### Create Users (POST)

```bash
# Request 1: Create first user
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

# Expected: 201 Created
# Response:
# {
#   "id": "507f1f77bcf86cd799439011",
#   "firstName": "Alice",
#   "lastName": "Johnson",
#   ...
# }
```

#### Get All Users (GET)

```bash
curl http://localhost:8080/api/users

# Expected: 200 OK
# Response: Array of all users
# [
#   {
#     "id": "507f1f77bcf86cd799439011",
#     "firstName": "Alice",
#     ...
#   }
# ]
```

#### Get Single User (GET)

```bash
# Replace ID with actual user id from previous response
curl http://localhost:8080/api/users/507f1f77bcf86cd799439011

# Expected: 200 OK
# Response: Single user object
```

#### Update User (PUT)

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

# Expected: 200 OK
# Response: Updated user object
```

#### Delete User (DELETE)

```bash
curl -X DELETE http://localhost:8080/api/users/507f1f77bcf86cd799439011

# Expected: 204 No Content
# Response: (empty)
```

#### Search by City (GET)

```bash
curl "http://localhost:8080/api/users/search/city?city=Los%20Angeles"

# Expected: 200 OK
# Response: Array of users in Los Angeles
```

#### Search by Age Range (GET)

```bash
curl "http://localhost:8080/api/users/search/age?minAge=20&maxAge=30"

# Expected: 200 OK
# Response: Array of users aged 20-30
```

---

## ✅ Verify Everything Works

Complete this checklist to ensure full setup:

```
Setup Verification Checklist:
============================

MongoDB Setup:
☐ MongoDB installed (brew install mongodb-community)
☐ MongoDB service running (brew services start mongodb-community)
☐ Can connect to MongoDB (mongosh works)
☐ Database visible in MongoDB (mongosh → show dbs)

Project Setup:
☐ Project compiles (./mvnw clean compile succeeds)
☐ All Java files present (src/main/java/org/acme/*.java)
☐ configuration correct (src/main/resources/application.properties)

Application Running:
☐ Quarkus dev mode started (./mvnw quarkus:dev)
☐ Application listening on 8080 (Listening on: http://localhost:8080)
☐ No errors in application logs

API Testing:
☐ Create user works (POST returns 201)
☐ Get all users works (GET returns array)
☐ Get single user works (GET {id} returns object)
☐ Update user works (PUT returns updated object)
☐ Delete user works (DELETE returns 204)
☐ Search by city works (GET search/city)
☐ Search by age works (GET search/age)

Postman Setup:
☐ Postman installed
☐ Collection imported
☐ All 10 requests visible
☐ Sample requests working
```

---

## 🆘 Troubleshooting

### Problem: MongoDB Connection Failed

**Error:** `MongoClientException: Unable to connect to server`

**Solution:**
1. Verify MongoDB is running: `brew services list | grep mongodb-community`
2. If not running: `brew services start mongodb-community`
3. Check MongoDB logs: `brew services log mongodb-community`
4. Verify connection string in `application.properties`

---

### Problem: Port 8080 Already in Use

**Error:** `Error: Failed to bind to 0.0.0.0/0.0.0.0:8080`

**Solution:**
```bash
# Find what's using port 8080
lsof -i :8080

# Kill the process (replace XXXX with PID)
kill -9 XXXX

# Or change Quarkus port in application.properties:
# quarkus.http.port=8081
```

---

### Problem: Java Version Too Old

**Error:** `java version "11.0.X" or lower`

**Solution:**
```bash
# Install Java 17 with Homebrew
brew install java17

# Set default Java version
export PATH="/usr/local/opt/openjdk@17/bin:$PATH"

# Verify
java -version
```

---

### Problem: Maven Build Fails

**Error:** `BUILD FAILURE` or dependency download errors

**Solution:**
```bash
# Clear Maven cache and rebuild
./mvnw clean install -U

# This removes old cache and re-downloads dependencies
```

---

### Problem: Postman Requests Failing

**Error:** `Connection refused` or `404 Not Found`

**Solution:**
1. Verify application is running: `curl http://localhost:8080/api/users`
2. Check request URL is correct
3. Check request method (POST vs GET)
4. Verify JSON format in body
5. Check Content-Type header is `application/json`

---

### Problem: MongoDB Compass Can't Connect

**Error:** `Connection timeout` or `Authentication failed`

**Solution:**
1. Verify MongoDB is running: `brew services list`
2. Try connecting via mongosh first: `mongosh`
3. If mongosh fails, MongoDB service isn't running
4. Restart MongoDB: `brew services restart mongodb-community`

---

## 🎨 MongoDB Compass (Optional)

MongoDB Compass is a visual GUI for exploring and managing MongoDB data.

### Installation

```bash
# Install via Homebrew
brew install --cask mongodb-compass

# Or download from: https://www.mongodb.com/try/download/compass
```

### First Connection

1. Open MongoDB Compass
2. Click **New Connection**
3. Connection string: `mongodb://localhost:27017`
4. Click **Connect**
5. In left sidebar, you'll see `quarkus_users` database
6. Expand it to see `users` collection
7. Click `users` to see all user documents

### Viewing Data

1. Click on any document to expand and view all fields
2. Use search box to filter documents
3. Click on field headers to sort

### Editing Data (Optional)

1. Double-click any value to edit
2. Click **Update** to save
3. Click **Delete** icon to remove document

---

## 🚢 Deployment

### Prepare for Production

```bash
# Build production JAR
./mvnw clean package -DskipTests

# Build takes ~2 minutes first time
# Creates: target/quarkus-app/quarkus-run.jar
```

### Run Production Build Locally

```bash
java -jar target/quarkus-app/quarkus-run.jar
```

### Deploy to Cloud (AWS/Azure/Google Cloud)

1. **Docker Container**
   ```bash
   # Create Dockerfile
   docker build -f src/main/docker/Dockerfile.jvm -t quarkus-api:latest .
   
   # Push to container registry
   docker tag quarkus-api:latest myregistry/quarkus-api:latest
   docker push myregistry/quarkus-api:latest
   ```

2. **Deploy to Kubernetes**
   - Use generated Docker image
   - Create Kubernetes deployment YAML
   - Deploy with kubectl

3. **Deploy to Cloud Platforms**
   - AWS: ECS, EKS, or AppRunner
   - Azure: App Service or AKS
   - Google Cloud: Cloud Run or GKE

---

## 📚 Configuration Reference

### application.properties Settings

```properties
# MongoDB Connection
quarkus.mongodb.connection-string=mongodb://localhost:27017
quarkus.mongodb.database=quarkus_users

# Application Info
quarkus.application.name=Quarkus Users API
quarkus.application.version=1.0.0

# REST Configuration
quarkus.rest.produces=application/json
quarkus.rest.consumes=application/json

# Port (default: 8080)
quarkus.http.port=8080

# Dev Mode
quarkus.dev.instrumentation=false
```

### Modify Settings

1. Open: `src/main/resources/application.properties`
2. Edit desired settings
3. Save file
4. Quarkus automatically reloads (in dev mode)

---

## 🔗 Quick Command Reference

```bash
# Start MongoDB
brew services start mongodb-community

# Stop MongoDB
brew services stop mongodb-community

# Check MongoDB status
brew services list

# Connect to MongoDB
mongosh

# Build project
./mvnw clean compile

# Run in dev mode
./mvnw quarkus:dev

# Build for production
./mvnw clean package -DskipTests

# Run production build
java -jar target/quarkus-app/quarkus-run.jar

# Test API
curl http://localhost:8080/api/users

# View logs
tail -f ~/.config/brew/Logs/mongodb-community/mongo.log
```

---

## 📖 Next Steps

1. ✅ **Setup Complete** - You have Quarkus running with MongoDB
2. 📖 **Read** `PROJECT_OVERVIEW.md` - Understand the architecture
3. 📚 **Review** `API_DOCUMENTATION.md` - Learn all endpoints
4. 🧪 **Test** - Use Postman to test all APIs
5. 🎨 **Optional** - Explore data with MongoDB Compass
6. 🚀 **Deploy** - Push to cloud when ready

---

## ❓ FAQ

**Q: Do I need Docker?**
A: No, Homebrew installation works fine on macOS. Docker is optional.

**Q: Can I use Windows or Linux?**
A: Yes! Installation steps are similar. Use apt-get (Linux) or Chocolatey (Windows).

**Q: Where are databases stored?**
A: Homebrew stores MongoDB data in `/usr/local/var/mongodb/`

**Q: What if MongoDB port 27017 is already used?**
A: Change MongoDB port or kill the process using it: `lsof -i :27017`

**Q: How do I backup my data?**
A: Use MongoDB Compass to export collection as JSON, or use `mongodump` command.

**Q: Can I use this API in production?**
A: Yes! Add authentication, validation, and error handling before deploying.

---

**Last Updated:** November 5, 2025  
**Quarkus Version:** 3.29.0  
**MongoDB Version:** 8.2.1  
**Java Version:** 17 LTS
