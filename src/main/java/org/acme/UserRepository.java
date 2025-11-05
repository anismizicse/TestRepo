package org.acme;

import io.quarkus.mongodb.panache.PanacheMongoRepository;
import jakarta.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class UserRepository implements PanacheMongoRepository<User> {
    
    // Additional query methods can be added here
    // Panache provides basic CRUD operations automatically:
    // - persist() -> Create
    // - find().by* -> Read
    // - update() -> Update
    // - delete() -> Delete
}
