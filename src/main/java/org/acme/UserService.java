package org.acme;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import java.util.List;
import java.util.Optional;
import org.bson.types.ObjectId;

@ApplicationScoped
public class UserService {
    
    @Inject
    UserRepository userRepository;
    
    /**
     * Create a new user
     */
    public User createUser(User user) {
        userRepository.persist(user);
        return user;
    }
    
    /**
     * Get all users
     */
    public List<User> getAllUsers() {
        return userRepository.listAll();
    }
    
    /**
     * Get user by ID
     */
    public Optional<User> getUserById(String id) {
        return userRepository.findByIdOptional(new ObjectId(id));
    }
    
    /**
     * Update an existing user
     */
    public User updateUser(String id, User user) {
        Optional<User> existingUser = userRepository.findByIdOptional(new ObjectId(id));
        if (existingUser.isPresent()) {
            User userToUpdate = existingUser.get();
            userToUpdate.firstName = user.firstName;
            userToUpdate.lastName = user.lastName;
            userToUpdate.email = user.email;
            userToUpdate.phoneNumber = user.phoneNumber;
            userToUpdate.city = user.city;
            userToUpdate.age = user.age;
            userRepository.update(userToUpdate);
            return userToUpdate;
        }
        return null;
    }
    
    /**
     * Delete a user by ID
     */
    public boolean deleteUser(String id) {
        return userRepository.deleteById(new ObjectId(id));
    }
    
    /**
     * Get users by city
     */
    public List<User> getUsersByCity(String city) {
        return userRepository.find("city", city).list();
    }
    
    /**
     * Get users by age range
     */
    public List<User> getUsersByAgeRange(int minAge, int maxAge) {
        return userRepository.find("age >= ?1 and age <= ?2", minAge, maxAge).list();
    }
}
