import torch

def normalize(vector,epsilon=1e-12):
    norm=torch.linalg.vector_norm(vector)
    return vector / norm.clamp_min(epsilon)

def directional_derivative(function, point, direction, epsilon=1e-3):
    """Approximation centrée de J_function(point) @ direction."""
    with torch.no_grad():
        output_plus = function(point + epsilon * direction)
        output_minus = function(point - epsilon * direction)

    return (output_plus - output_minus) / (2.0 * epsilon)


def power_method_q(model , image ,sigma , n_iter=5, epsilon=1e-3):
    model.eval()
    image = image.detach().clone().requires_grad_(True)
    
    def Q_function(input_image):
        return 2*model(input_image,sigma)-input_image
    
    vector_v=torch.randn_like(image)
    vector_v=normalize(vector_v)
    for _ in range(n_iter):
        # u = J_Q(x) @ v, par différences finies
        vector_u = directional_derivative(
            Q_function,
            image,
            vector_v,
            epsilon,
        )
        vector_u = normalize(vector_u)

        # v = J_Q(x).T @ u
        q_output = Q_function(image)

        scalar_product = torch.sum(
            q_output * vector_u.detach()
        )

        vector_v = torch.autograd.grad(
            scalar_product,
            image,
            create_graph=False,
            retain_graph=False,
        )[0]

        vector_v = normalize(vector_v)

    # Estimation finale de sigma_max(J_Q)
    jacobian_vector = directional_derivative(
        Q_function,
        image,
        vector_v,
        epsilon,
    )

    return torch.linalg.vector_norm(jacobian_vector)