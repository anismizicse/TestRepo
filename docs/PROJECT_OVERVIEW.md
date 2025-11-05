# Quarkus MongoDB CRUD API - Project Overview

## 📚 Welcome Beginner Developers!

This is a complete **Quarkus microservice** that demonstrates how to build a modern REST API with MongoDB database integration. This document explains the entire project structure, architecture, and how everything works together.

---

## 🎯 What Is This Project?

A **production-ready REST API** built with Quarkus that:
- ✅ Creates, reads, updates, and deletes (CRUD) user data
- ✅ Stores data in MongoDB (NoSQL database)
- ✅ Follows best practices with layered architecture
- ✅ Provides search and filtering capabilities
- ✅ Can be tested with Postman
- ✅ Runs on Java 17+ with minimal startup time

**Real-World Use Case:** An application backend that manages user profiles with search functionality.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT (Browser/Postman)             │
│              Sends HTTP requests to API                 │
└──────────────────────┬────────────────────────────────┘
                       │ HTTP (REST API calls)
                       ▼
┌─────────────────────────────────────────────────────────┐
│          QUARKUS APPLICATION (Java Framework)           │
│             Runs on http://localhost:8080               │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │  API LAYER: UserResource                        │   │
│  │  Receives HTTP requests, returns JSON           │   │
│  │  Routes: POST, GET, PUT, DELETE, SEARCH         │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │  SERVICE LAYER: UserService                     │   │
│  │  Business logic, data validation, operations    │   │
│  │  Processes requests, applies rules              │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │  REPOSITORY LAYER: UserRepository               │   │
│  │  Data access, database queries                  │   │
│  │  Handles persistence operations                 │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │  ENTITY MODEL: User                             │   │
│  │  Java class representing database document      │   │
│  └────────────────────┬────────────────────────────┘   │
└────────────────────────┬─────────────────────────────┘
                         │ MongoDB Driver (Panache)
                         │ TCP/IP Connection
                         ▼
┌─────────────────────────────────────────────────────────┐
│           MONGODB DATABASE (localhost:27017)            │
│  Database: quarkus_users                                │
│  Collection: users                                      │
│  Documents: User records as JSON-like objects           │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
getting-started/
├── src/
│   ├── main/
│   │   ├── java/org/acme/
│   │   │   ├── User.java              ← Data model (Entity)
│   │   │   ├── UserRepository.java     ← Data access layer
│   │   │   ├── UserService.java        ← Business logic layer
│   │   │   └── UserResource.java       ← REST API endpoints
│   │   └── resources/
│   │       └── application.properties  ← Configuration file
│   │
│   └── test/
│       └── java/org/acme/
│           ├── GreetingResourceTest.java
│           └── GreetingResourceIT.java
│
├── pom.xml                             ← Maven build configuration
├── mvnw / mvnw.cmd                     ← Maven wrapper (build tool)
├── README.md                           ← Quick start guide
├── Quarkus_Users_API.postman_collection.json ← API testing file
│
└── docs/
    ├── PROJECT_OVERVIEW.md             ← This file (beginner guide)
    ├── SETUP_GUIDE.md                  ← Installation & configuration
    ├── API_DOCUMENTATION.md            ← All API endpoints
    ├── PUBLISH_QUICK_GUIDE.md          ← Share API with team
    └── COMPASS_SIMPLE_GUIDE.md         ← View data in MongoDB GUI
```

---

## 🔑 Core Components Explained

### 1. **User.java** (Entity - Data Model)

**What it does:** Defines the structure of user data that gets stored in MongoDB.

```java
public class User extends PanacheMongoEntity {
    public String firstName;
    public String lastName;
    public String email;
    public String phoneNumber;
    public String city;
    public int age;
}
```

**Key Points:**
- Extends `PanacheMongoEntity` (Quarkus abstraction for MongoDB)
- Each field becomes a property in the MongoDB document
- `id` field is automatically generated as MongoDB ObjectId
- `public` fields allow direct access (no getters/setters needed)

**Database Representation:**
```json
{
  "_id": "507f1f77bcf86cd799439011",
  "firstName": "John",
  "lastName": "Doe",
  "email": "john.doe@example.com",
  "phoneNumber": "+1-555-0123",
  "city": "New York",
  "age": 28
}
```

---

### 2. **UserRepository.java** (Data Access Layer)

**What it does:** Handles all database queries and persistence operations. Acts as a bridge between application and MongoDB.

```java
@ApplicationScoped
public class UserRepository implements PanacheMongoRepository<User> {
    // No code needed! Panache provides all CRUD methods automatically
}
```

**Automatic Methods Provided:**
| Method | Purpose |
|--------|---------|
| `persist(user)` | Save new user to database |
| `findByIdOptional(id)` | Get user by ID, returns Optional (safe null handling) |
| `listAll()` | Retrieve all users |
| `update(user)` | Update existing user |
| `deleteById(id)` | Delete user by ID |
| `find(query, params)` | Run custom queries |

**Example Usage (in UserService):**
```java
userRepository.persist(user);  // Create
userRepository.findByIdOptional(id);  // Read
userRepository.update(user);  // Update
userRepository.deleteById(id);  // Delete
```

---

### 3. **UserService.java** (Business Logic Layer)

**What it does:** Contains business logic, validation, and coordinates between API layer and repository.

```java
@ApplicationScoped
public class UserService {
    @Inject UserRepository userRepository;
    
    public void createUser(User user) { ... }
    public List<User> getAllUsers() { ... }
    public Optional<User> getUserById(String id) { ... }
    public void updateUser(String id, User user) { ... }
    public boolean deleteUser(String id) { ... }
    public List<User> getUsersByCity(String city) { ... }
    public List<User> getUsersByAgeRange(int minAge, int maxAge) { ... }
}
```

**Key Methods:**
- **createUser()** - Validates and saves new user
- **getAllUsers()** - Returns all users from database
- **getUserById()** - Finds specific user, returns Optional for safe null handling
- **updateUser()** - Modifies existing user
- **deleteUser()** - Removes user from database
- **getUsersByCity()** - Searches users by city name
- **getUsersByAgeRange()** - Filters users by age range

**Why Separate Service?** 
Keeps business logic separate from REST endpoints, making code reusable and testable.

---

### 4. **UserResource.java** (REST API Layer)

**What it does:** Exposes HTTP endpoints that clients use to interact with the API.

```java
@Path("/api/users")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class UserResource {
    @Inject UserService userService;
    
    @POST public Response createUser(User user) { ... }
    @GET public List<User> getAllUsers() { ... }
    @GET @Path("/{id}") public User getUserById(@PathParam("id") String id) { ... }
    @PUT @Path("/{id}") public Response updateUser(...) { ... }
    @DELETE @Path("/{id}") public Response deleteUser(...) { ... }
}
```

**HTTP Endpoints Exposed:**
```
POST   /api/users                      Create user
GET    /api/users                      Get all users
GET    /api/users/{id}                 Get single user
PUT    /api/users/{id}                 Update user
DELETE /api/users/{id}                 Delete user
GET    /api/users/search/city?...      Search by city
GET    /api/users/search/age?...       Search by age
```

**How REST Works:**
- Client sends HTTP request (e.g., `GET /api/users`)
- Quarkus routes it to appropriate method
- Method processes request and calls service
- Response is automatically converted to JSON
- JSON sent back to client

---

## 🔄 Request Flow - Step by Step

### Example: Create a New User

```
1. CLIENT (Postman) sends HTTP POST request:
   POST http://localhost:8080/api/users
   Body: {"firstName": "John", "lastName": "Doe", ...}
   ↓
2. QUARKUS receives request and routes to UserResource.createUser()
   ↓
3. RESOURCE (@Path annotation) extracts JSON and converts to User object
   ↓
4. SERVICE (UserService) validates data:
   - Check fields not empty
   - Check email format
   ↓
5. REPOSITORY (UserRepository) calls persist():
   - Converts User object to MongoDB document
   ↓
6. MONGODB stores document in "users" collection
   - Auto-generates _id field
   - Stores with timestamp
   ↓
7. REPOSITORY returns saved user with new _id
   ↓
8. SERVICE returns user to Resource
   ↓
9. RESOURCE converts User to JSON
   ↓
10. QUARKUS sends HTTP 201 response with JSON:
    {
      "id": "507f1f77bcf86cd799439011",
      "firstName": "John",
      ...
    }
    ↓
11. POSTMAN/CLIENT receives and displays response
```

---

## 📊 Layered Architecture Benefits

```
┌─────────────────────────────────────────────┐
│  API LAYER (UserResource)                   │
│  • Handles HTTP requests/responses           │
│  • REST endpoint definitions                 │
│  • Input/output serialization                │
│  → Problem: Hard to test HTTP parts          │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│  SERVICE LAYER (UserService)                │
│  • Business logic                            │
│  • Data validation                           │
│  • Business rules                            │
│  → Problem: Reusable logic                   │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│  REPOSITORY LAYER (UserRepository)          │
│  • Database operations                       │
│  • Query logic                               │
│  • Persistence                               │
│  → Problem: Database specific               │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│  MONGODB                                    │
│  • Persistent data storage                  │
│  • Document-oriented database               │
└─────────────────────────────────────────────┘

Benefits of this layering:
✅ Each layer has single responsibility
✅ Easy to test each layer independently
✅ Easy to modify database without changing API
✅ Easy to change business logic
✅ Follows industry best practices
```

---

## ⚙️ Technology Stack Explained

### **Quarkus 3.29.0**
- **What:** Modern Java framework for microservices
- **Why:** Fast startup, low memory, perfect for cloud/containers
- **Features:** Hot reload (live changes), dev mode, production ready

### **MongoDB 8.2.1**
- **What:** NoSQL database that stores JSON-like documents
- **Why:** Flexible schema, natural for Java objects, easy to use
- **Comparison:** Unlike SQL databases with rigid tables, MongoDB is more flexible

### **Java 17 LTS**
- **What:** Programming language version
- **Why:** Latest stable, long-term support, modern features

### **Maven**
- **What:** Build tool for compiling and packaging Java code
- **Why:** Manages dependencies, handles compilation, builds runnable JAR

### **Panache ODM**
- **What:** Quarkus abstraction for MongoDB
- **Why:** Reduces boilerplate code, provides repository pattern automatically

### **RESTEasy & Jackson**
- **What:** REST API framework and JSON serialization
- **Why:** Automatic HTTP routing and JSON conversion

---

## 📋 Database Concepts

### What is MongoDB?

MongoDB is a **NoSQL database** that stores data as documents (similar to JSON):

**SQL Database (Traditional):**
```
Table: users
┌─────┬──────────┬──────────┬─────────────────────────────┐
│ id  │ name     │ email    │ address                     │
├─────┼──────────┼──────────┼─────────────────────────────┤
│ 1   │ John     │ j@ex.com │ 123 Main St, New York, NY   │
│ 2   │ Jane     │ ja@ex.com│ 456 Oak Ave, Boston, MA     │
└─────┴──────────┴──────────┴─────────────────────────────┘
```

**MongoDB Collection (NoSQL):**
```json
db.users = [
  {
    "_id": ObjectId(...),
    "firstName": "John",
    "lastName": "Doe",
    "email": "john@example.com",
    "phoneNumber": "+1-555-0123",
    "city": "New York",
    "age": 28
  },
  {
    "_id": ObjectId(...),
    "firstName": "Jane",
    "lastName": "Smith",
    "email": "jane@example.com",
    "phoneNumber": "+1-555-0456",
    "city": "Boston",
    "age": 26
  }
]
```

**Key Differences:**
| Feature | SQL | MongoDB |
|---------|-----|---------|
| Schema | Rigid tables | Flexible documents |
| Format | Rows/columns | JSON-like objects |
| Scaling | Vertical | Horizontal |
| Joins | Complex | Nested documents |

---

## 🚀 How Quarkus Dev Mode Works

When you run `./mvnw quarkus:dev`:

1. **Compilation** - Code compiled to Java bytecode
2. **Server Start** - Application starts on port 8080
3. **File Watching** - Quarkus watches for file changes
4. **Live Reload** - Change code → Quarkus recompiles automatically
5. **No Restart Needed** - See changes instantly (most of the time)

**Benefits:**
- Instant feedback while developing
- No need to restart server after code changes
- Fast iteration cycle
- Perfect for testing API changes

---

## 📊 Data Flow Diagram

```
USER ACTION              QUARKUS PROCESSING           RESULT
═════════════════════════════════════════════════════════════════

User clicks "Create"
    │
    ▼
HTTP POST request
    │
    ▼
Quarkus routes to UserResource.createUser()
    │
    ▼
Deserialize JSON → User object
    │
    ▼
Inject UserService
    │
    ▼
UserService.createUser(user)
    │
    ▼
Validate user data
    │
    ├─ Invalid? → Return 400 Bad Request
    │
    └─ Valid? → Continue
       │
       ▼
    Inject UserRepository
       │
       ▼
    UserRepository.persist(user)
       │
       ▼
    MongoDB saves document
       │
       ▼
    MongoDB generates _id
       │
       ▼
    Return User with _id
       │
       ▼
    Serialize to JSON
       │
       ▼
    HTTP 201 Created response
       │
       ▼
    User sees success message
```

---

## 🎓 Learning Path

### As a Beginner Developer, Follow This Order:

1. **Understand REST APIs**
   - HTTP methods: GET (read), POST (create), PUT (update), DELETE (delete)
   - Endpoints: `/api/users`, `/api/users/{id}`
   - Response codes: 200 (OK), 201 (Created), 404 (Not Found), 500 (Error)

2. **Learn about Databases**
   - MongoDB vs SQL databases
   - Document-oriented storage
   - Collections and documents
   - CRUD operations

3. **Understand Layered Architecture**
   - Why separate API → Service → Repository?
   - Each layer has responsibility
   - Makes testing easier

4. **Study Java Concepts**
   - Object-oriented programming
   - Dependency injection (@Inject)
   - Annotations (@Path, @POST, etc.)
   - Exception handling

5. **Explore the Code**
   - Start with User.java (entity)
   - Read UserRepository.java (data access)
   - Study UserService.java (business logic)
   - Analyze UserResource.java (API endpoints)

6. **Test the API**
   - Use Postman to make requests
   - Try creating users
   - Retrieve and search users
   - Update and delete users

---

## 🔗 Key Java Concepts Used

### Dependency Injection
```java
@Inject UserRepository userRepository;
```
Automatically creates instance of UserRepository and provides it. Alternative to `new UserRepository()`.

### Annotations
```java
@Path("/api/users")      // URL path for this class
@POST                    // HTTP POST method
@GET                     // HTTP GET method
@ApplicationScoped       // Single instance for entire app
@Inject                  // Inject dependency
```
Metadata that tells Quarkus what to do with the class/method.

### Generics
```java
public class UserRepository implements PanacheMongoRepository<User>
```
`<User>` means this repository handles User objects specifically.

### Optional
```java
Optional<User> user = userRepository.findByIdOptional(id);
```
Safe way to handle values that might not exist (instead of returning null).

---

## 📚 File Locations Quick Reference

| Component | File | Purpose |
|-----------|------|---------|
| Entity | `src/main/java/org/acme/User.java` | Data model |
| Repository | `src/main/java/org/acme/UserRepository.java` | Database access |
| Service | `src/main/java/org/acme/UserService.java` | Business logic |
| Resource | `src/main/java/org/acme/UserResource.java` | REST endpoints |
| Config | `src/main/resources/application.properties` | MongoDB connection |
| Tests | `src/test/java/org/acme/` | Test files |
| Build | `pom.xml` | Project configuration |
| API Testing | `Quarkus_Users_API.postman_collection.json` | Postman requests |

---

## 💡 Quick Concepts

### What is REST?
**RE**presentational **S**tate **T**ransfer - A way to design APIs using HTTP methods:
- **GET** - Fetch data
- **POST** - Create data
- **PUT** - Update data  
- **DELETE** - Remove data

### What is JSON?
JavaScript Object Notation - A text format for representing data:
```json
{
  "firstName": "John",
  "lastName": "Doe",
  "age": 28
}
```

### What is HTTP Status Codes?
Numbers that indicate what happened:
- **200** - OK (success)
- **201** - Created (new resource created)
- **400** - Bad Request (client error)
- **404** - Not Found (resource doesn't exist)
- **500** - Server Error

### What is ObjectId?
MongoDB's unique identifier for documents:
```
507f1f77bcf86cd799439011
```
Auto-generated, ensures no duplicates.

---

## 📖 Next Steps

1. **Read** `SETUP_GUIDE.md` - Install and run the application
2. **Review** `API_DOCUMENTATION.md` - Learn all available endpoints
3. **Follow** `POSTMAN_GUIDE.md` in SETUP_GUIDE.md - Test API with Postman
4. **Explore** MongoDB Compass - Visualize your database
5. **Modify** the code - Add your own features

---

## ❓ Common Questions

**Q: Why use Quarkus instead of Spring Boot?**
A: Quarkus is faster, uses less memory, and is better for microservices and containers.

**Q: Can I use SQL instead of MongoDB?**
A: Yes! Replace Panache ODM with Panache ORM. Concepts remain similar.

**Q: Why Layered Architecture?**
A: Makes code testable, maintainable, and follows industry best practices.

**Q: What does @ApplicationScoped mean?**
A: One instance of this class exists for the entire application lifecycle.

**Q: How is data persisted in MongoDB?**
A: When you call `persist()`, data is converted to BSON and stored as a document.

**Q: Can this run on cloud?**
A: Yes! Package as Docker container and deploy to AWS, Azure, Google Cloud, etc.

---

## 🎯 Summary

This Quarkus project demonstrates:
- ✅ Modern REST API design
- ✅ NoSQL database integration (MongoDB)
- ✅ Layered architecture best practices
- ✅ Dependency injection patterns
- ✅ CRUD operations
- ✅ Search/filter functionality
- ✅ Professional Java coding standards
- ✅ Production-ready microservice architecture

**You now understand the complete flow from HTTP request to database storage and back!**

---

**Last Updated:** November 5, 2025  
**Quarkus Version:** 3.29.0  
**MongoDB Version:** 8.2.1  
**Java Version:** 17 LTS
