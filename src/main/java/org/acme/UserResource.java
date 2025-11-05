package org.acme;

import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.validation.ConstraintViolationException;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Path("/api/users")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class UserResource {
    
    @Inject
    UserService userService;
    
    /**
     * POST - Create a new user
     * POST /api/users
     */
    @POST
    public Response createUser(@Valid User user) {
        User createdUser = userService.createUser(user);
        return Response.status(Response.Status.CREATED).entity(createdUser).build();
    }
    
    /**
     * GET - Retrieve all users
     * GET /api/users
     */
    @GET
    public Response getAllUsers() {
        List<User> users = userService.getAllUsers();
        return Response.ok(users).build();
    }
    
    /**
     * GET - Retrieve a specific user by ID
     * GET /api/users/{id}
     */
    @GET
    @Path("/{id}")
    public Response getUserById(@PathParam("id") String id) {
        try {
            Optional<User> user = userService.getUserById(id);
            if (user.isPresent()) {
                return Response.ok(user.get()).build();
            } else {
                return Response.status(Response.Status.NOT_FOUND)
                        .entity("User not found with id: " + id)
                        .build();
            }
        } catch (IllegalArgumentException e) {
            return Response.status(Response.Status.BAD_REQUEST)
                    .entity("Invalid user ID format")
                    .build();
        }
    }
    
    /**
     * PUT - Update an existing user
     * PUT /api/users/{id}
     */
    @PUT
    @Path("/{id}")
    public Response updateUser(@PathParam("id") String id, @Valid User user) {
        try {
            User updatedUser = userService.updateUser(id, user);
            if (updatedUser != null) {
                return Response.ok(updatedUser).build();
            } else {
                return Response.status(Response.Status.NOT_FOUND)
                        .entity("User not found with id: " + id)
                        .build();
            }
        } catch (IllegalArgumentException e) {
            return Response.status(Response.Status.BAD_REQUEST)
                    .entity("Invalid user ID format")
                    .build();
        }
    }
    
    /**
     * DELETE - Remove a user
     * DELETE /api/users/{id}
     */
    @DELETE
    @Path("/{id}")
    public Response deleteUser(@PathParam("id") String id) {
        try {
            boolean deleted = userService.deleteUser(id);
            if (deleted) {
                return Response.noContent().build();
            } else {
                return Response.status(Response.Status.NOT_FOUND)
                        .entity("User not found with id: " + id)
                        .build();
            }
        } catch (IllegalArgumentException e) {
            return Response.status(Response.Status.BAD_REQUEST)
                    .entity("Invalid user ID format")
                    .build();
        }
    }
    
    /**
     * GET - Search users by city
     * GET /api/users/search/city?city=NewYork
     */
    @GET
    @Path("/search/city")
    public Response getUsersByCity(@QueryParam("city") String city) {
        if (city == null || city.isEmpty()) {
            return Response.status(Response.Status.BAD_REQUEST)
                    .entity("City parameter is required")
                    .build();
        }
        List<User> users = userService.getUsersByCity(city);
        return Response.ok(users).build();
    }
    
    /**
     * GET - Search users by age range
     * GET /api/users/search/age?minAge=20&maxAge=30
     */
    @GET
    @Path("/search/age")
    public Response getUsersByAgeRange(
            @QueryParam("minAge") int minAge,
            @QueryParam("maxAge") int maxAge) {
        if (minAge < 0 || maxAge < 0 || minAge > maxAge) {
            return Response.status(Response.Status.BAD_REQUEST)
                    .entity("Invalid age range parameters")
                    .build();
        }
        List<User> users = userService.getUsersByAgeRange(minAge, maxAge);
        return Response.ok(users).build();
    }
}
