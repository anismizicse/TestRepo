# Quarkus MongoDB CRUD API# getting-started



A production-ready REST API built with **Quarkus** and **MongoDB** demonstrating modern microservice architecture patterns.This project uses Quarkus, the Supersonic Subatomic Java Framework.



**Includes:** CRUD operations, search functionality, layered architecture, and complete documentation.If you want to learn more about Quarkus, please visit its website: <https://quarkus.io/>.



---## Running the application in dev mode



## 🎯 Quick Start (3 Steps)You can run your application in dev mode that enables live coding using:



### 1. Setup MongoDB```shell script

```bash./mvnw quarkus:dev

# Install MongoDB (macOS with Homebrew)```

brew tap mongodb/brew && brew install mongodb-community

brew services start mongodb-community> **_NOTE:_**  Quarkus now ships with a Dev UI, which is available in dev mode only at <http://localhost:8080/q/dev/>.

```

## Packaging and running the application

### 2. Start Application

```bashThe application can be packaged using:

./mvnw quarkus:dev

``````shell script

Application runs at: **http://localhost:8080/api/users**./mvnw package

```

### 3. Test API

```bashIt produces the `quarkus-run.jar` file in the `target/quarkus-app/` directory.

# Create userBe aware that it’s not an _über-jar_ as the dependencies are copied into the `target/quarkus-app/lib/` directory.

curl -X POST http://localhost:8080/api/users \

  -H "Content-Type: application/json" \The application is now runnable using `java -jar target/quarkus-app/quarkus-run.jar`.

  -d '{"firstName":"John","lastName":"Doe","email":"john@example.com","phoneNumber":"+1-555-0000","city":"New York","age":28}'

If you want to build an _über-jar_, execute the following command:

# Get all users

curl http://localhost:8080/api/users```shell script

```./mvnw package -Dquarkus.package.jar.type=uber-jar

```

---

The application, packaged as an _über-jar_, is now runnable using `java -jar target/*-runner.jar`.

## 📚 Documentation

## Creating a native executable

Complete guides for understanding and using this project:

You can create a native executable using:

| Document | Purpose | For Whom |

|----------|---------|----------|```shell script

| **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** | Understand architecture, components, concepts | Beginners learning Quarkus |./mvnw package -Dnative

| **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** | Installation, configuration, troubleshooting | Getting started |```

| **[API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)** | All endpoints, parameters, examples | API consumers |

| **[PUBLISH_QUICK_GUIDE.md](docs/PUBLISH_QUICK_GUIDE.md)** | Share API with team via Postman | Team collaboration |Or, if you don't have GraalVM installed, you can run the native executable build in a container using:

| **[COMPASS_SIMPLE_GUIDE.md](docs/COMPASS_SIMPLE_GUIDE.md)** | Visual data management with MongoDB | Data exploration |

```shell script

---./mvnw package -Dnative -Dquarkus.native.container-build=true

```

## 🏗️ Project Structure

You can then execute your native executable with: `./target/getting-started-1.0.0-SNAPSHOT-runner`

```

src/main/If you want to learn more about building native executables, please consult <https://quarkus.io/guides/maven-tooling>.

├── java/org/acme/

│   ├── User.java              ← Data model (Entity)## Related Guides

│   ├── UserRepository.java     ← Data access layer

│   ├── UserService.java        ← Business logic- REST ([guide](https://quarkus.io/guides/rest)): A Jakarta REST implementation utilizing build time processing and Vert.x. This extension is not compatible with the quarkus-resteasy extension, or any of the extensions that depend on it.

│   └── UserResource.java       ← REST endpoints

└── resources/## Provided Code

    └── application.properties  ← Configuration

### REST

docs/

├── PROJECT_OVERVIEW.md         ← Beginner's guideEasily start your REST Web Services

├── SETUP_GUIDE.md              ← Installation & config

├── API_DOCUMENTATION.md        ← API reference[Related guide section...](https://quarkus.io/guides/getting-started-reactive#reactive-jax-rs-resources)

├── PUBLISH_QUICK_GUIDE.md      ← Share with team
└── COMPASS_SIMPLE_GUIDE.md     ← Data management

pom.xml                          ← Dependencies & build
```

---

## 🚀 Available Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| **POST** | `/api/users` | Create user |
| **GET** | `/api/users` | Get all users |
| **GET** | `/api/users/{id}` | Get user by ID |
| **PUT** | `/api/users/{id}` | Update user |
| **DELETE** | `/api/users/{id}` | Delete user |
| **GET** | `/api/users/search/city?city=X` | Search by city |
| **GET** | `/api/users/search/age?minAge=X&maxAge=Y` | Search by age |

**Full API documentation:** See [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)

---

## 🔑 Core Technologies

- **Quarkus 3.29.0** - Modern Java microservices framework
- **MongoDB 8.2.1** - NoSQL document database
- **Java 17 LTS** - Programming language
- **Maven 3.9.11** - Build tool
- **RESTEasy** - REST implementation
- **Jackson** - JSON serialization
- **Panache ODM** - MongoDB abstraction layer

---

## 💻 Development Commands

```bash
# Build project
./mvnw clean compile

# Run in dev mode (live reload)
./mvnw quarkus:dev

# Run tests
./mvnw test

# Build for production
./mvnw clean package -DskipTests

# Run production build
java -jar target/quarkus-app/quarkus-run.jar

# Build native executable
./mvnw package -Dnative
```

---

## 📋 Data Model

Each user has these fields:

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

---

## ⚙️ Architecture

**Layered architecture** for clean separation of concerns:

```
REST API Layer     → Handles HTTP requests/responses
  ↓
Business Logic     → Service layer with business rules
  ↓
Data Access        → Repository pattern for DB operations
  ↓
MongoDB Database   → Persistent storage
```

**Benefits:**
- ✅ Easy to test each layer independently
- ✅ Simple to change database later
- ✅ Clean, maintainable code
- ✅ Follows industry best practices

---

## 🧪 Testing

### Using Postman

1. Import `Quarkus_Users_API.postman_collection.json` into Postman
2. Run requests from collection
3. See pre-configured examples with sample data

### Using curl

```bash
# Create
curl -X POST http://localhost:8080/api/users \
  -H "Content-Type: application/json" \
  -d '{"firstName":"Alice","lastName":"Johnson","email":"alice@example.com","phoneNumber":"+1-555-7890","city":"LA","age":25}'

# Read all
curl http://localhost:8080/api/users

# Read one
curl http://localhost:8080/api/users/507f1f77bcf86cd799439011

# Update
curl -X PUT http://localhost:8080/api/users/507f1f77bcf86cd799439011 \
  -H "Content-Type: application/json" \
  -d '{"firstName":"Alice","lastName":"Williams","email":"alice.w@example.com","phoneNumber":"+1-555-9999","city":"SF","age":26}'

# Delete
curl -X DELETE http://localhost:8080/api/users/507f1f77bcf86cd799439011

# Search city
curl "http://localhost:8080/api/users/search/city?city=New%20York"

# Search age
curl "http://localhost:8080/api/users/search/age?minAge=25&maxAge=35"
```

---

## 📊 API Response Status Codes

| Status | Meaning | Example |
|--------|---------|---------|
| **200** | OK | GET, PUT successful |
| **201** | Created | POST successful |
| **204** | No Content | DELETE successful |
| **400** | Bad Request | Invalid input |
| **404** | Not Found | User doesn't exist |
| **500** | Server Error | Database error |

---

## 🔐 Security Notes

**Current:** No authentication (development)

**For production, add:**
- ✅ JWT token authentication
- ✅ HTTPS/TLS encryption
- ✅ Input validation & sanitization
- ✅ Rate limiting
- ✅ CORS policy
- ✅ API key management

---

## 🛠️ Configuration

MongoDB connection (in `application.properties`):
```properties
quarkus.mongodb.connection-string=mongodb://localhost:27017
quarkus.mongodb.database=quarkus_users
```

Change port:
```properties
quarkus.http.port=8081
```

---

## 📖 Learn More

- **Quarkus:** https://quarkus.io
- **MongoDB:** https://www.mongodb.com
- **REST APIs:** https://restfulapi.net
- **Layered Architecture:** https://en.wikipedia.org/wiki/Multitier_architecture

---

## 📁 Documentation Quick Links

**For Different Audiences:**

👶 **I'm new to Quarkus** → Start with [PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)

🚀 **I want to setup and run** → Follow [SETUP_GUIDE.md](docs/SETUP_GUIDE.md)

📡 **I need API details** → Read [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)

🔄 **I want to share APIs** → Check [PUBLISH_QUICK_GUIDE.md](docs/PUBLISH_QUICK_GUIDE.md)

💾 **I want to view/edit data** → See [COMPASS_SIMPLE_GUIDE.md](docs/COMPASS_SIMPLE_GUIDE.md)

---

## ✅ Verify Setup Works

```bash
# 1. MongoDB running?
mongosh --eval "db.version()"

# 2. Start application
./mvnw quarkus:dev

# 3. API responding?
curl http://localhost:8080/api/users

# Should return: []
```

---

## 🎓 Next Steps

1. **Read** documentation for your use case (links above)
2. **Follow** SETUP_GUIDE.md to get everything running
3. **Test** endpoints with provided Postman collection
4. **Explore** source code to understand patterns
5. **Modify** for your use case

---

## 📝 License

MIT License - See LICENSE file

---

## 🤝 Contributing

Improvements welcome! Fork, modify, and submit pull requests.

---

**Version:** 1.0.0  
**Last Updated:** November 5, 2025  
**Status:** Production Ready ✅
