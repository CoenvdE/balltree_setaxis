#ifndef BALLTREE_FIXED_H
#define BALLTREE_FIXED_H

/**
 * @file balltree.h
 * @brief Implementation of Ball Tree data structure for batch processing
 * 
 * This implementation creates complete binary trees for batches of points.
 * Each leaf node contains either one point (duplicated) or two points.
 * The tree is built recursively by selecting split dimensions based on
 * maximum spread of points along each dimension.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>

using DataType = double;
using IndexType = long;
using MaskType = bool;

// ============================================================================
// Global PCA Alignment Functions
// ============================================================================

// Constants for power iteration
const int MAX_POWER_ITERATIONS = 100;
const DataType CONVERGENCE_THRESHOLD = 1e-10;

/**
 * @brief Compute centroid of a batch of points
 * 
 * @param data          Pointer to data array (num_points × num_features)
 * @param num_points    Number of points
 * @param num_features  Number of features per point
 * @param centroid      Output array for centroid (size num_features)
 */
void compute_batch_centroid(
    const DataType* data,
    IndexType num_points,
    IndexType num_features,
    DataType* centroid
) {
    // Initialize centroid to zero
    for (IndexType d = 0; d < num_features; ++d) {
        centroid[d] = 0.0;
    }
    
    // Sum all points
    for (IndexType i = 0; i < num_points; ++i) {
        for (IndexType d = 0; d < num_features; ++d) {
            centroid[d] += data[i * num_features + d];
        }
    }
    
    // Divide by number of points
    for (IndexType d = 0; d < num_features; ++d) {
        centroid[d] /= num_points;
    }
}

/**
 * @brief Compute covariance matrix for a batch of points
 * 
 * @param data          Pointer to data array (num_points × num_features)
 * @param num_points    Number of points
 * @param num_features  Number of features per point
 * @param centroid      Centroid of the points (size num_features)
 * @param cov           Output covariance matrix (size num_features × num_features, row-major)
 */
void compute_batch_covariance(
    const DataType* data,
    IndexType num_points,
    IndexType num_features,
    const DataType* centroid,
    DataType* cov
) {
    // Initialize covariance to zero
    for (IndexType i = 0; i < num_features * num_features; ++i) {
        cov[i] = 0.0;
    }
    
    // Compute covariance: sum of (x - mean)(x - mean)^T
    for (IndexType i = 0; i < num_points; ++i) {
        for (IndexType d1 = 0; d1 < num_features; ++d1) {
            DataType diff1 = data[i * num_features + d1] - centroid[d1];
            for (IndexType d2 = 0; d2 < num_features; ++d2) {
                DataType diff2 = data[i * num_features + d2] - centroid[d2];
                cov[d1 * num_features + d2] += diff1 * diff2;
            }
        }
    }
}

/**
 * @brief Find principal eigenvector of 2x2 symmetric matrix (closed-form)
 * 
 * Uses coordinate-free tie-breaking for degenerate cases by selecting
 * the direction to the furthest point from centroid.
 * 
 * @param data          Pointer to data array (num_points × num_features)
 * @param num_points    Number of points
 * @param num_features  Number of features (must be 2)
 * @param centroid     Centroid of the points (size 2)
 * @param cov           2x2 covariance matrix (row-major)
 * @param eigenvec      Output eigenvector (size 2)
 * @return              true if valid direction found, false if degenerate
 */
bool principal_eigenvector_2d(
    const DataType* data,
    IndexType num_points,
    IndexType num_features,
    const DataType* centroid,
    const DataType* cov,
    DataType* eigenvec
) {
    DataType a = cov[0], b = cov[1], c = cov[3];
    DataType trace = a + c;
    DataType discriminant_sq = (a - c) * (a - c) / 4.0 + b * b;
    DataType discriminant = std::sqrt(discriminant_sq);
    
    // Check if eigenvalues are degenerate (equal)
    DataType eigenvalue_diff = 2.0 * discriminant;
    DataType eigenvalue_scale = std::abs(trace) / 2.0 + 1e-10;
    
    if (eigenvalue_diff < 1e-7 * eigenvalue_scale) {
        // DEGENERATE CASE: Use the furthest point as the axis (coordinate-free)
        DataType max_dist_sq = -1.0;
        for (IndexType i = 0; i < num_points; ++i) {
            DataType dx = data[i * num_features + 0] - centroid[0];
            DataType dy = data[i * num_features + 1] - centroid[1];
            DataType d2 = dx * dx + dy * dy;
            if (d2 > max_dist_sq) {
                max_dist_sq = d2;
                eigenvec[0] = dx;
                eigenvec[1] = dy;
            }
        }
        DataType norm = std::sqrt(eigenvec[0] * eigenvec[0] + eigenvec[1] * eigenvec[1]);
        if (norm > 1e-12) {
            eigenvec[0] /= norm;
            eigenvec[1] /= norm;
        } else {
            // All points at centroid - use default
            eigenvec[0] = 1.0;
            eigenvec[1] = 0.0;
            return false;
        }
        return true;
    }
    
    DataType lambda1 = trace / 2.0 + discriminant;

    if (std::abs(b) > 1e-12) {
        eigenvec[0] = b;
        eigenvec[1] = lambda1 - a;
    } else {
        // When b is zero, we still need coordinate-free tie-breaking
        if (std::abs(a - c) < 1e-10) {
            // Equal eigenvalues - use furthest point
            DataType max_dist_sq = -1.0;
            for (IndexType i = 0; i < num_points; ++i) {
                DataType dx = data[i * num_features + 0] - centroid[0];
                DataType dy = data[i * num_features + 1] - centroid[1];
                DataType d2 = dx * dx + dy * dy;
                if (d2 > max_dist_sq) {
                    max_dist_sq = d2;
                    eigenvec[0] = dx;
                    eigenvec[1] = dy;
                }
            }
            DataType norm = std::sqrt(eigenvec[0] * eigenvec[0] + eigenvec[1] * eigenvec[1]);
            if (norm > 1e-12) {
                eigenvec[0] /= norm;
                eigenvec[1] /= norm;
            } else {
                eigenvec[0] = 1.0;
                eigenvec[1] = 0.0;
                return false;
            }
        } else if (a >= c) {
            eigenvec[0] = 1.0;
            eigenvec[1] = 0.0;
        } else {
            eigenvec[0] = 0.0;
            eigenvec[1] = 1.0;
        }
    }

    // Normalize
    DataType norm = std::sqrt(eigenvec[0] * eigenvec[0] + eigenvec[1] * eigenvec[1]);
    if (norm > 1e-12) {
        eigenvec[0] /= norm;
        eigenvec[1] /= norm;
    }
    return true;
}

/**
 * @brief Find principal eigenvector of 3x3 symmetric matrix using power iteration
 * 
 * Uses rotation-equivariant initialization: the column of the covariance matrix
 * with the largest norm.
 * 
 * @param cov       3x3 covariance matrix (row-major)
 * @param eigenvec  Output eigenvector (size 3)
 * @return          true if valid direction found, false if degenerate
 */
bool principal_eigenvector_3d(
    const DataType* cov,
    DataType* eigenvec
) {
    // Rotation-equivariant initialization: use the column with largest norm
    DataType col_norms[3];
    for (int j = 0; j < 3; ++j) {
        col_norms[j] = 0.0;
        for (int i = 0; i < 3; ++i) {
            col_norms[j] += cov[i * 3 + j] * cov[i * 3 + j];
        }
    }
    
    int best_col = 0;
    if (col_norms[1] > col_norms[0]) best_col = 1;
    if (col_norms[2] > col_norms[best_col]) best_col = 2;
    
    // Initialize v with the best column (normalized)
    DataType v[3], v_new[3];
    DataType init_norm = std::sqrt(col_norms[best_col]);
    if (init_norm < 1e-12) {
        // Degenerate case: zero matrix
        eigenvec[0] = 1.0; eigenvec[1] = 0.0; eigenvec[2] = 0.0;
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        v[i] = cov[i * 3 + best_col] / init_norm;
    }
    
    // Power iteration
    for (int iter = 0; iter < MAX_POWER_ITERATIONS; ++iter) {
        // v_new = A @ v
        for (int i = 0; i < 3; ++i) {
            v_new[i] = 0.0;
            for (int j = 0; j < 3; ++j) {
                v_new[i] += cov[i * 3 + j] * v[j];
            }
        }
        
        // Normalize
        DataType norm = std::sqrt(v_new[0]*v_new[0] + v_new[1]*v_new[1] + v_new[2]*v_new[2]);
        if (norm < 1e-12) {
            eigenvec[0] = 1.0; eigenvec[1] = 0.0; eigenvec[2] = 0.0;
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            v_new[i] /= norm;
        }
        
        // Check convergence
        DataType diff = 0.0;
        for (int i = 0; i < 3; ++i) {
            diff += (v_new[i] - v[i]) * (v_new[i] - v[i]);
        }
        
        // Copy v_new to v
        for (int i = 0; i < 3; ++i) {
            v[i] = v_new[i];
        }
        
        if (diff < CONVERGENCE_THRESHOLD) {
            break;
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        eigenvec[i] = v[i];
    }
    return true;
}

/**
 * @brief Apply sign disambiguation using higher-order moments
 * 
 * Ensures consistent sign of the principal direction across rotations.
 * Uses third moment (skewness) as primary signal, with fallback to
 * fifth moment for symmetric distributions. This is order-invariant
 * and works for all point arrangements.
 * 
 * @param data          Pointer to data array
 * @param num_points    Number of points
 * @param num_features  Number of features per point
 * @param centroid      Centroid of the points
 * @param direction     Principal direction (will be flipped if needed)
 */
void disambiguate_sign(
    const DataType* data,
    IndexType num_points,
    IndexType num_features,
    const DataType* centroid,
    DataType* direction
) {
    DataType skewness = 0.0;
    DataType fifth_moment = 0.0;  // Higher moments are even more stable for symmetry

    for (IndexType i = 0; i < num_points; ++i) {
        DataType proj = 0.0;
        for (IndexType d = 0; d < num_features; ++d) {
            proj += (data[i * num_features + d] - centroid[d]) * direction[d];
        }
        DataType p3 = proj * proj * proj;
        skewness += p3;
        fifth_moment += p3 * proj * proj;
    }
    
    // If skewness is zero (symmetric), use the 5th moment. 
    // If both are zero, the shape is so symmetric the sign truly doesn't matter,
    // but the sum ensures we aren't dependent on point ordering.
    DataType decision_signal = (std::abs(skewness) > 1e-9) ? skewness : fifth_moment;
    
    if (decision_signal < 0) {
        for (IndexType d = 0; d < num_features; ++d) {
            direction[d] = -direction[d];
        }
    }
}

/**
 * @brief Compute the alignment rotation matrix for a batch of points
 * 
 * Computes a rotation matrix that aligns the principal axis with the x-axis.
 * This makes the tree construction rotation-invariant.
 * 
 * @param data              Pointer to data array (num_points × num_features)
 * @param num_points        Number of points
 * @param num_features      Number of features per point (2 or 3)
 * @param rotation_matrix   Output rotation matrix (num_features × num_features, row-major)
 */
void compute_alignment_rotation(
    const DataType* data,
    IndexType num_points,
    IndexType num_features,
    DataType* rotation_matrix
) {
    // Initialize rotation matrix to identity
    for (IndexType i = 0; i < num_features * num_features; ++i) {
        rotation_matrix[i] = 0.0;
    }
    for (IndexType i = 0; i < num_features; ++i) {
        rotation_matrix[i * num_features + i] = 1.0;
    }
    
    if (num_points <= 1) {
        return;  // Keep identity for trivial case
    }
    
    // Compute centroid
    DataType centroid[3] = {0.0, 0.0, 0.0};
    compute_batch_centroid(data, num_points, num_features, centroid);
    
    // Compute covariance matrix
    DataType cov[9] = {0.0};  // Max 3x3
    compute_batch_covariance(data, num_points, num_features, centroid, cov);
    
    // Find principal eigenvector
    DataType direction[3] = {1.0, 0.0, 0.0};
    bool valid = true;
    
    if (num_features == 2) {
        valid = principal_eigenvector_2d(data, num_points, num_features, centroid, cov, direction);
    } else if (num_features == 3) {
        valid = principal_eigenvector_3d(cov, direction);
    }
    
    if (!valid) {
        // Degenerate case - use identity rotation
        return;
    }
    
    // Apply sign disambiguation
    disambiguate_sign(data, num_points, num_features, centroid, direction);
    
    // Build rotation matrix that rotates 'direction' to [1, 0, ...] (x-axis)
    if (num_features == 2) {
        // For 2D: R rotates direction to [1, 0]
        // R = [[d_x, d_y], [-d_y, d_x]]
        DataType dx = direction[0];
        DataType dy = direction[1];
        rotation_matrix[0] = dx;   rotation_matrix[1] = dy;
        rotation_matrix[2] = -dy;  rotation_matrix[3] = dx;
    } else if (num_features == 3) {
        // For 3D: Build orthonormal basis with direction as first vector
        DataType e1[3] = {direction[0], direction[1], direction[2]};
        
        // Find e2: The principal direction in the subspace perpendicular to e1
        // We use the point furthest from the e1 axis as initial guess
        DataType e2[3] = {0.0, 0.0, 0.0};
        
        // Initial guess for e2: vector to the point furthest from the e1 axis
        DataType max_d2 = -1.0;
        for (IndexType i = 0; i < num_points; ++i) {
            DataType p[3];
            for(int d=0; d<3; ++d) p[d] = data[i*3+d] - centroid[d];
            DataType proj = p[0]*e1[0] + p[1]*e1[1] + p[2]*e1[2];
            DataType perp[3] = {p[0] - proj*e1[0], p[1] - proj*e1[1], p[2] - proj*e1[2]};
            DataType d2 = perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2];
            if (d2 > max_d2) {
                max_d2 = d2;
                for(int d=0; d<3; ++d) e2[d] = perp[d];
            }
        }
        
        DataType norm_e2 = std::sqrt(e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2]);
        if (norm_e2 < 1e-12) { // Perfectly collinear points
            e2[0] = -e1[1]; e2[1] = e1[0]; e2[2] = 0; // Fallback
            if (std::abs(e1[2]) > 0.9) { e2[0] = 1; e2[1] = 0; e2[2] = 0; }
            norm_e2 = std::sqrt(e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2]);
        }
        e2[0] /= norm_e2; e2[1] /= norm_e2; e2[2] /= norm_e2;

        // Disambiguate e2 sign using moments
        disambiguate_sign(data, num_points, num_features, centroid, e2);
        
        // e3 is fixed by right-hand rule
        DataType e3[3] = {
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0]
        };
        
        // Full equivariant rotation matrix
        rotation_matrix[0] = e1[0]; rotation_matrix[1] = e1[1]; rotation_matrix[2] = e1[2];
        rotation_matrix[3] = e2[0]; rotation_matrix[4] = e2[1]; rotation_matrix[5] = e2[2];
        rotation_matrix[6] = e3[0]; rotation_matrix[7] = e3[1]; rotation_matrix[8] = e3[2];
    }
}

/**
 * @brief Apply alignment rotation to transform data points
 * 
 * Transforms points: aligned[i] = R @ data[i]
 * 
 * @param data              Input data array (num_points × num_features)
 * @param num_points        Number of points
 * @param num_features      Number of features per point
 * @param rotation_matrix   Rotation matrix (num_features × num_features, row-major)
 * @param aligned_data      Output aligned data array (num_points × num_features)
 */
void apply_alignment(
    const DataType* data,
    IndexType num_points,
    IndexType num_features,
    const DataType* rotation_matrix,
    DataType* aligned_data
) {
    for (IndexType i = 0; i < num_points; ++i) {
        for (IndexType d = 0; d < num_features; ++d) {
            aligned_data[i * num_features + d] = 0.0;
            for (IndexType k = 0; k < num_features; ++k) {
                aligned_data[i * num_features + d] += 
                    rotation_matrix[d * num_features + k] * data[i * num_features + k];
            }
        }
    }
}

// ============================================================================
// Original Axis-Aligned Tree Functions
// ============================================================================

/**
 * @brief Find the dimension with maximum spread for splitting points
 * 
 * @param data          Pointer to the data array (num_points × num_features)
 * @param idx_array     Array of indices into data
 * @param start         Start index in idx_array
 * @param end           End index in idx_array (exclusive)
 * @param num_features  Number of features per point
 * @return              Index of the dimension with maximum spread
 */
IndexType find_split_dimension(
    const DataType* data,
    const IndexType* idx_array,
    IndexType start,
    IndexType end,
    IndexType num_features
) {
    const IndexType num_points = end - start;
    if (num_points <= 1) return 0;  // Handle edge case

    IndexType best_dim = 0;
    DataType max_spread = 0;
    
    // Find dimension with maximum spread
    for (IndexType dim = 0; dim < num_features; ++dim) {
        // Initialize with first point's value
        DataType min_val = data[idx_array[start] * num_features + dim];
        DataType max_val = min_val;
        
        // Find min and max values along this dimension
        for (IndexType i = start + 1; i < end; ++i) {
            DataType val = data[idx_array[i] * num_features + dim];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        const DataType spread = max_val - min_val;
        if (spread > max_spread) {
            max_spread = spread;
            best_dim = dim;
        }
    }
    
    return best_dim;
}

/**
 * @brief Partition points around median value along given dimension
 * 
 * Uses std::nth_element to efficiently partition points such that
 * the point at split_index is in its sorted position, with smaller
 * values to the left and larger values to the right.
 * 
 * @param data          Pointer to data array
 * @param node_indices  Array of indices to partition
 * @param split_dim     Dimension along which to split
 * @param split_index   Position of the split
 * @param num_features  Number of features per point
 * @param num_points    Number of points to partition
 */
void partition_node_indices(
    const DataType* data,
    IndexType* node_indices,
    IndexType split_dim,
    IndexType split_index,
    IndexType num_features,
    IndexType num_points
) {
    // Lambda for comparing points along split dimension
    auto compare = [data, split_dim, num_features](IndexType a, IndexType b) {
        return data[a * num_features + split_dim] < data[b * num_features + split_dim];
    };
    
    std::nth_element(
        node_indices,
        node_indices + split_index,
        node_indices + num_points,
        compare
    );
}

/**
 * @brief Recursively build a ball tree
 * 
 * Creates a complete binary tree by recursively splitting points
 * along dimensions with maximum spread. Leaf nodes contain either
 * one point (duplicated) or two points.
 * 
 * @param data           Pointer to data array
 * @param idx_array      Working array of indices
 * @param output_indices Output array for tree structure
 * @param output_mask    Output array indicating dummy duplicates
 * @param idx_start      Start index in idx_array
 * @param idx_end        End index in idx_array (exclusive)
 * @param output_start   Start index in output_indices
 * @param num_features   Number of features per point
 * @param current_level  Current depth in tree
 * @param max_level      Maximum tree depth
 */
void build_tree(
    const DataType* data,
    IndexType* idx_array,
    IndexType* output_indices,
    MaskType* output_mask,
    IndexType idx_start,
    IndexType idx_end,
    IndexType output_start,
    IndexType num_features,
    IndexType current_level,
    IndexType max_level
) {
    const IndexType num_points = idx_end - idx_start;
    
    // Handle leaf nodes
    if (current_level == max_level) {
        if (num_points == 1) {
            // Single point - duplicate it
            const IndexType idx = idx_array[idx_start];
            output_indices[output_start] = idx;
            output_indices[output_start + 1] = idx;
            output_mask[output_start] = true;      // Original point
            output_mask[output_start + 1] = false; // Duplicate
        } else {
            // Two or more points - take first two
            output_indices[output_start] = idx_array[idx_start];
            output_indices[output_start + 1] = idx_array[idx_start + 1];
            output_mask[output_start] = true;     // Both points are original
            output_mask[output_start + 1] = true;
        }
        return;
    }
    
    // Find best splitting dimension and partition points
    const IndexType split_dim = find_split_dimension(
        data, idx_array, idx_start, idx_end, num_features
    );
    const IndexType mid_point = (idx_start + idx_end) / 2;
    
    if (num_points > 1) {
        partition_node_indices(
            data,
            idx_array + idx_start,
            split_dim,
            mid_point - idx_start,
            num_features,
            num_points
        );
    }
    
    // Calculate output position for left child's points
    const IndexType left_size = (1 << (max_level - current_level - 1)) * 2;
    
    // Recursively build left and right subtrees
    build_tree(
        data, idx_array, output_indices, output_mask,
        idx_start, mid_point, output_start,
        num_features, current_level + 1, max_level
    );
    
    build_tree(
        data, idx_array, output_indices, output_mask,
        mid_point, idx_end, output_start + left_size,
        num_features, current_level + 1, max_level
    );
}


/**
 * @brief Partitions leaves of a complete ball tree K times again assuming the leaves were subjected to an orthogonal transformation
 * 
 * Takes leaves of a complete ball tree (power of 2 size) and partitions them
 * K times, resulting in partitions of size N/2^K.
 * 
 * @param data              Pointer to data array
 * @param idx_array         Working array of indices (initialized to [0,1,2,...])
 * @param output_indices    Output array for the partitioned indices
 * @param idx_start         Start index in idx_array
 * @param idx_end           End index in idx_array (exclusive)
 * @param output_start      Start index in output_indices
 * @param num_features      Number of features per point
 * @param current_level   How many times data was partitioned
 * @param target_level    Number of partitioning steps
 */
void partition_ball_tree(
    const DataType* data,
    IndexType* idx_array,
    IndexType* output_indices,
    IndexType idx_start,
    IndexType idx_end,
    IndexType output_start,
    IndexType num_features,
    IndexType current_level,
    IndexType target_level
) {
    const IndexType num_points = idx_end - idx_start;
    
    // If we've reached the desired partitioning level or have only one point, stop recursion
    if (target_level == current_level || num_points <= 1) {
        // Copy the indices to the output
        for (IndexType i = 0; i < num_points; ++i) {
            output_indices[output_start + i] = idx_array[idx_start + i];
        }
        return;
    }
    
    // Find best splitting dimension and partition points
    const IndexType split_dim = find_split_dimension(
        data, idx_array, idx_start, idx_end, num_features
    );
    const IndexType mid_point = (idx_start + idx_end) / 2;
    
    if (num_points > 1) {
        partition_node_indices(
            data,
            idx_array + idx_start,
            split_dim,
            mid_point - idx_start,
            num_features,
            num_points
        );
    }
    
    // Calculate output position for left child's points
    const IndexType left_size = mid_point - idx_start;
    
    // Recursively build left and right subtrees
    partition_ball_tree(
        data, idx_array, output_indices,
        idx_start, mid_point, output_start,
        num_features, current_level + 1, target_level
    );
    
    partition_ball_tree(
        data, idx_array, output_indices,
        mid_point, idx_end, output_start + left_size,
        num_features, current_level + 1, target_level
    );
}


#endif // BALLTREE_FIXED_H