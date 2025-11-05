package org.acme;

import io.quarkus.mongodb.panache.PanacheMongoEntity;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import jakarta.validation.constraints.Pattern;

public class User extends PanacheMongoEntity {
    
    @NotBlank(message = "First name is required and cannot be empty")
    @Size(min = 1, max = 50, message = "First name must be between 1 and 50 characters")
    public String firstName;
    
    @NotBlank(message = "Last name is required and cannot be empty")
    @Size(min = 1, max = 50, message = "Last name must be between 1 and 50 characters")
    public String lastName;
    
    @NotBlank(message = "Email is required and cannot be empty")
    @Email(message = "Email must be a valid email address")
    public String email;
    
    @NotBlank(message = "Phone number is required and cannot be empty")
    @Pattern(regexp = "^[+]?[0-9\\-\\s\\(\\)]{8,20}$", message = "Phone number must be valid (8-20 characters, can include +, -, (), spaces)")
    public String phoneNumber;
    
    @NotBlank(message = "City is required and cannot be empty")
    @Size(min = 1, max = 50, message = "City must be between 1 and 50 characters")
    public String city;
    
    @NotNull(message = "Age is required")
    @Min(value = 0, message = "Age must be 0 or greater")
    @Max(value = 150, message = "Age must be 150 or less")
    public int age;

    // Default constructor
    public User() {
    }

    // Constructor with parameters
    public User(String firstName, String lastName, String email, String phoneNumber, String city, int age) {
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.phoneNumber = phoneNumber;
        this.city = city;
        this.age = age;
    }

    // Getters and Setters
    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", firstName='" + firstName + '\'' +
                ", lastName='" + lastName + '\'' +
                ", email='" + email + '\'' +
                ", phoneNumber='" + phoneNumber + '\'' +
                ", city='" + city + '\'' +
                ", age=" + age +
                '}';
    }
}
