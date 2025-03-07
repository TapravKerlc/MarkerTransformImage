import os
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import slicer
from DICOMLib import DICOMUtils
from collections import deque
import vtk
from slicer.ScriptedLoadableModule import *
import qt
import matplotlib.pyplot as plt
import csv

#exec(open("C:/Users/lkomar/Documents/Prostata/FirstTryRegister.py").read())

class SeekTransformModule(ScriptedLoadableModule):
    """
    Module description shown in the module panel.
    """
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Seek Transform module"
        self.parent.categories = ["Image Processing"]
        self.parent.contributors = ["Luka Komar (Onkološki Inštitut Ljubljana, Fakulteta za Matematiko in Fiziko Ljubljana)"]
        self.parent.helpText = "This module applies rigid transformations to CBCT volumes based on reference CT volumes."
        self.parent.acknowledgementText = "Supported by doc. Primož Peterlin & prof. Andrej Studen"

class SeekTransformModuleWidget(ScriptedLoadableModuleWidget):
    """
    GUI of the module.
    """
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Dropdown menu za izbiro metode
        self.rotationMethodComboBox = qt.QComboBox()
        self.rotationMethodComboBox.addItems(["Kabsch", "Horn", "Iterative Closest Point (Kabsch)"])
        self.layout.addWidget(self.rotationMethodComboBox)

        # Checkboxi za transformacije
        self.rotationCheckBox = qt.QCheckBox("Rotation")
        self.rotationCheckBox.setChecked(True)
        self.layout.addWidget(self.rotationCheckBox)

        self.translationCheckBox = qt.QCheckBox("Translation")
        self.translationCheckBox.setChecked(True)
        self.layout.addWidget(self.translationCheckBox)

        self.scalingCheckBox = qt.QCheckBox("Scaling")
        self.scalingCheckBox.setChecked(True)
        self.layout.addWidget(self.scalingCheckBox)
        
        self.writefileCheckBox = qt.QCheckBox("Write distances to csv file")
        self.writefileCheckBox.setChecked(True)
        self.layout.addWidget(self.writefileCheckBox)

        # Load button
        self.applyButton = qt.QPushButton("Find markers and transform")
        self.applyButton.toolTip = "Finds markers, computes optimal rigid transform and applies it to CBCT volumes."
        self.applyButton.enabled = True
        self.layout.addWidget(self.applyButton)

        # Connect button to logic
        self.applyButton.connect('clicked(bool)', self.onApplyButton)

        self.layout.addStretch(1)

    def onApplyButton(self):
        logic = MyTransformModuleLogic()
        selectedMethod = self.rotationMethodComboBox.currentText  # izberi metodo izračuna rotacije

        # Preberi stanje checkboxov
        applyRotation = self.rotationCheckBox.isChecked()
        applyTranslation = self.translationCheckBox.isChecked()
        applyScaling = self.scalingCheckBox.isChecked()
        writefilecheck = self.writefileCheckBox.isChecked()

        # Pokliči logiko z izbranimi nastavitvami
        logic.run(selectedMethod, applyRotation, applyTranslation, applyScaling, writefilecheck)


class MyTransformModuleLogic(ScriptedLoadableModuleLogic):
    """
    Core logic of the module.
    """
    
    
    def run(self, selectedMethod, applyRotation, applyTranslation, applyScaling, writefilecheck):
        print("Calculating...")
        
        def group_points(points, threshold):
            # Function to group points that are close to each other
            grouped_points = []
            while points:
                point = points.pop()  # Take one point from the list
                group = [point]  # Start a new group
                
                # Find all points close to this one
                distances = cdist([point], points)  # Calculate distances from this point to others
                close_points = [i for i, dist in enumerate(distances[0]) if dist < threshold]
                
                # Add the close points to the group
                group.extend([points[i] for i in close_points])
                
                # Remove the grouped points from the list
                points = [point for i, point in enumerate(points) if i not in close_points]
                
                # Add the group to the result
                grouped_points.append(group)
            
            return grouped_points

        def region_growing(image_data, seed, intensity_threshold, max_distance):
            dimensions = image_data.GetDimensions()
            visited = set()
            region = []
            queue = deque([seed])

            while queue:
                x, y, z = queue.popleft()
                if (x, y, z) in visited:
                    continue

                visited.add((x, y, z))
                voxel_value = image_data.GetScalarComponentAsDouble(x, y, z, 0)
                
                if voxel_value >= intensity_threshold:
                    region.append((x, y, z))
                    # Add neighbors within bounds
                    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < dimensions[0] and 0 <= ny < dimensions[1] and 0 <= nz < dimensions[2]:
                            if (nx, ny, nz) not in visited:
                                queue.append((nx, ny, nz))

            return region

        
        def compute_optimal_scaling_per_axis(moving_points, fixed_points):
            """Computes optimal scaling factors for each axis (X, Y, Z) to align moving points (CBCT) to fixed points (CT).

            Args:
                moving_points (list of lists): List of (x, y, z) moving points (CBCT).
                fixed_points (list of lists): List of (x, y, z) fixed points (CT).

            Returns:
                tuple: Scaling factors (sx, sy, sz).
            """
            moving_points_np = np.array(moving_points)
            fixed_points_np = np.array(fixed_points)

            # Compute centroids
            centroid_moving = np.mean(moving_points_np, axis=0)
            centroid_fixed = np.mean(fixed_points_np, axis=0)

            # Compute absolute distances of each point from its centroid along each axis
            distances_moving = np.abs(moving_points_np - centroid_moving)
            distances_fixed = np.abs(fixed_points_np - centroid_fixed)

            # Compute scaling factors as the ratio of mean absolute distances per axis
            scale_factors = np.mean(distances_fixed, axis=0) / np.mean(distances_moving, axis=0)

            return tuple(scale_factors)

        def compute_scaling(cbct_points, scaling_factors):
            """Applies non-uniform scaling to CBCT points.

            Args:
                cbct_points (list of lists): List of (x, y, z) points.
                scaling_factors (tuple): Scaling factors (sx, sy, sz) for each axis.

            Returns:
                np.ndarray: Scaled CBCT points.
            """
            sx, sy, sz = scaling_factors  # Extract scaling factors
            scaling_matrix = np.diag([sx, sy, sz])  # Create diagonal scaling matrix

            cbct_points_np = np.array(cbct_points)  # Convert to numpy array
            scaled_points = cbct_points_np @ scaling_matrix.T  # Apply scaling

            return scaled_points.tolist()  # Convert back to list

        def compute_Kabsch_rotation(moving_points, fixed_points):
            """
            Computes the optimal rotation matrix to align moving_points to fixed_points.
            
            Parameters:
            moving_points (list or ndarray): List of points to be rotated CBCT
            fixed_points (list or ndarray): List of reference points CT

            Returns:
            ndarray: Optimal rotation matrix.
            """
            assert len(moving_points) == len(fixed_points), "Point lists must be the same length."

            # Convert to numpy arrays
            moving = np.array(moving_points)
            fixed = np.array(fixed_points)

            # Compute centroids
            centroid_moving = np.mean(moving, axis=0)
            centroid_fixed = np.mean(fixed, axis=0)

            # Center the points
            moving_centered = moving - centroid_moving
            fixed_centered = fixed - centroid_fixed

            # Compute covariance matrix
            H = np.dot(moving_centered.T, fixed_centered)

            # SVD decomposition
            U, _, Vt = np.linalg.svd(H)
            Rotate_optimal = np.dot(Vt.T, U.T)

            # Correct improper rotation (reflection)
            if np.linalg.det(Rotate_optimal) < 0:
                Vt[-1, :] *= -1
                Rotate_optimal = np.dot(Vt.T, U.T)

            return Rotate_optimal


        def compute_Horn_rotation(moving_points, fixed_points):
            """
            Computes the optimal rotation matrix using quaternions.

            Parameters:
            moving_points (list or ndarray): List of points to be rotated.
            fixed_points (list or ndarray): List of reference points.

            Returns:
            ndarray: Optimal rotation matrix.
            """
            assert len(moving_points) == len(fixed_points), "Point lists must be the same length."
            
            moving = np.array(moving_points)
            fixed = np.array(fixed_points)
            
            # Compute centroids
            centroid_moving = np.mean(moving, axis=0)
            centroid_fixed = np.mean(fixed, axis=0)
            
            # Center the points
            moving_centered = moving - centroid_moving
            fixed_centered = fixed - centroid_fixed
            
            # Construct the cross-dispersion matrix
            M = np.dot(moving_centered.T, fixed_centered)
            
            # Construct the N matrix for quaternion solution
            A = M - M.T
            delta = np.array([A[1, 2], A[2, 0], A[0, 1]])
            trace = np.trace(M)
            
            N = np.zeros((4, 4))
            N[0, 0] = trace
            N[1:, 0] = delta
            N[0, 1:] = delta
            N[1:, 1:] = M + M.T - np.eye(3) * trace
            
            # Compute the eigenvector corresponding to the maximum eigenvalue
            eigvals, eigvecs = np.linalg.eigh(N)
            q_optimal = eigvecs[:, np.argmax(eigvals)]  # Optimal quaternion
            
            # Convert quaternion to rotation matrix
            w, x, y, z = q_optimal
            R = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
            ])
            
            return R

        def icp_algorithm(moving_points, fixed_points, max_iterations=100, tolerance=1e-5):
            """
            Iterative Closest Point (ICP) algorithm to align moving_points to fixed_points.
            
            Parameters:
            moving_points (list or ndarray): List of points to be aligned.
            fixed_points (list or ndarray): List of reference points.
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance.

            Returns:
            ndarray: Transformed moving points.
            ndarray: Optimal rotation matrix.
            ndarray: Optimal translation vector.
            """
            # Convert to numpy arrays
            moving = np.array(moving_points)
            fixed = np.array(fixed_points)

            # Initialize transformation
            R = np.eye(3)  # Identity matrix for rotation
            t = np.zeros(3)  # Zero vector for translation

            prev_error = np.inf  # Initialize previous error to a large value

            for iteration in range(max_iterations):
                # Step 1: Find the nearest neighbors (correspondences)
                distances = np.linalg.norm(moving[:, np.newaxis] - fixed, axis=2)
                nearest_indices = np.argmin(distances, axis=1)
                nearest_points = fixed[nearest_indices]

                # Step 2: Compute the optimal rotation and translation
                R_new = compute_Kabsch_rotation(moving, nearest_points)
                centroid_moving = np.mean(moving, axis=0)
                centroid_fixed = np.mean(nearest_points, axis=0)
                t_new = centroid_fixed - np.dot(R_new, centroid_moving)

                # Step 3: Apply the transformation
                moving = np.dot(moving, R_new.T) + t_new

                # Update the cumulative transformation
                R = np.dot(R_new, R)
                t = np.dot(R_new, t) + t_new

                # Step 4: Check for convergence
                mean_error = np.mean(np.linalg.norm(moving - nearest_points, axis=1))
                if np.abs(prev_error - mean_error) < tolerance:
                    print(f"ICP converged after {iteration + 1} iterations.")
                    break
                prev_error = mean_error

            else:
                print(f"ICP reached maximum iterations ({max_iterations}).")

            return moving, R, t


        def compute_translation(moving_points, fixed_points, rotation_matrix):
            """
            Computes the translation vector to align moving_points to fixed_points given a rotation matrix.
            
            Parameters:
            moving_points (list or ndarray): List of points to be translated.
            fixed_points (list or ndarray): List of reference points.
            rotation_matrix (ndarray): Rotation matrix.

            Returns:
            ndarray: Translation vector.
            """
            # Convert to numpy arrays
            moving = np.array(moving_points)
            fixed = np.array(fixed_points)

            # Compute centroids
            centroid_moving = np.mean(moving, axis=0)
            centroid_fixed = np.mean(fixed, axis=0)

            # Compute translation
            translation = centroid_fixed - np.dot(centroid_moving, rotation_matrix)

            return translation

        def create_vtk_transform(rotation_matrix, translation_vector): 
            """
            Creates a vtkTransform from a rotation matrix and a translation vector.
            """
            # Create a 4x4 transformation matrix
            transform_matrix = np.eye(4)  # Start with an identity matrix
            transform_matrix[:3, :3] = rotation_matrix  # Set rotation part
            transform_matrix[:3, 3] = translation_vector  # Set translation part

            # Convert to vtkMatrix4x4
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    vtk_matrix.SetElement(i, j, transform_matrix[i, j])
            #print("Transform matrix:")
            #for i in range(4):
            #    print(" ".join(f"{vtk_matrix.GetElement(i, j):.6f}" for j in range(4)))
            # Create vtkTransform and set the matrix
            transform = vtk.vtkTransform()
            transform.SetMatrix(vtk_matrix)
            return transform
        
        
        def detect_points_region_growing(volume_name, yesCbct, intensity_threshold=3000, x_min=90, x_max=380, y_min=190, y_max=380, z_min=80, z_max=140, max_distance=9, centroid_merge_threshold=5):
            volume_node = slicer.util.getNode(volume_name)
            if not volume_node:
                raise RuntimeError(f"Volume {volume_name} not found.")
            
            image_data = volume_node.GetImageData()
            matrix = vtk.vtkMatrix4x4()
            volume_node.GetIJKToRASMatrix(matrix)

            dimensions = image_data.GetDimensions()
            #detected_regions = []

            if yesCbct: #je cbct ali ct?
                valid_x_min, valid_x_max = 0, dimensions[0] - 1
                valid_y_min, valid_y_max = 0, dimensions[1] - 1
                valid_z_min, valid_z_max = 0, dimensions[2] - 1
            else:
                valid_x_min, valid_x_max = max(x_min, 0), min(x_max, dimensions[0] - 1)
                valid_y_min, valid_y_max = max(y_min, 0), min(y_max, dimensions[1] - 1)
                valid_z_min, valid_z_max = max(z_min, 0), min(z_max, dimensions[2] - 1)

            visited = set()

            def grow_region(x, y, z):
                if (x, y, z) in visited:
                    return None

                voxel_value = image_data.GetScalarComponentAsDouble(x, y, z, 0)
                if voxel_value < intensity_threshold:
                    return None

                region = region_growing(image_data, (x, y, z), intensity_threshold, max_distance=max_distance)
                if region:
                    for point in region:
                        visited.add(tuple(point))
                    return region
                return None

            regions = []
            for z in range(valid_z_min, valid_z_max + 1):
                for y in range(valid_y_min, valid_y_max + 1):
                    for x in range(valid_x_min, valid_x_max + 1):
                        region = grow_region(x, y, z)
                        if region:
                            regions.append(region)

            # Collect centroids using intensity-weighted average
            centroids = []
            for region in regions:
                points = np.array([matrix.MultiplyPoint([*point, 1])[:3] for point in region])
                intensities = np.array([image_data.GetScalarComponentAsDouble(*point, 0) for point in region])
                
                if intensities.sum() > 0:
                    weighted_centroid = np.average(points, axis=0, weights=intensities)
                    max_intensity = intensities.max()
                    centroids.append((np.round(weighted_centroid, 2), max_intensity))

            unique_centroids = []
            for centroid, intensity in centroids:
                if not any(np.linalg.norm(centroid - existing_centroid) < centroid_merge_threshold for existing_centroid, _ in unique_centroids):
                    unique_centroids.append((centroid, intensity))
                    
            markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"Markers_{volume_name}")
            for centroid, intensity in unique_centroids:
                markups_node.AddControlPoint(*centroid)
                #print(f"Detected Centroid (RAS): {centroid}, Max Intensity: {intensity}")

            return unique_centroids
    
        
        
        # Globalni seznami za končno statistiko
        prostate_size_est = []
        ctcbct_distance = []

        # Pridobimo SubjectHierarchyNode
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        
        studyItems = vtk.vtkIdList()
        shNode.GetItemChildren(shNode.GetSceneItemID(), studyItems)

        for i in range(studyItems.GetNumberOfIds()):
            studyItem = studyItems.GetId(i)
            
            # **LOKALNI** seznami, resetirajo se pri vsakem study-ju
            cbct_list = []
            ct_list = []
            volume_points_dict = {}

            # Get child items of the study item
            volumeItems = vtk.vtkIdList()
            shNode.GetItemChildren(studyItem, volumeItems)

            # Iteracija čez vse volumne v posameznem studyju
            for j in range(volumeItems.GetNumberOfIds()):
                intermediateItem = volumeItems.GetId(j)
    
                # Preveri, ali je to dejanska skupina volumnov (npr. "No study description")
                intermediateName = shNode.GetItemName(intermediateItem)
                #print(f"Checking intermediate item: {intermediateName}")

                finalVolumeItems = vtk.vtkIdList()
                shNode.GetItemChildren(intermediateItem, finalVolumeItems)  # Išči globlje!

                for k in range(finalVolumeItems.GetNumberOfIds()):
                    volumeItem = finalVolumeItems.GetId(k)
                    volumeNode = shNode.GetItemDataNode(volumeItem)
                    
                    dicomUIDs = volumeNode.GetAttribute("DICOM.instanceUIDs")
                    if not dicomUIDs:
                        print("❌ This is an NRRD volume!")
                        continue  # Preskoči, če ni DICOM volume
                    
                    if volumeNode and volumeNode.IsA("vtkMRMLScalarVolumeNode"):
                        print(f"✔️ Found volume: {volumeNode.GetName()} (ID: {volumeItem})")
                
                    if not volumeNode or not volumeNode.IsA("vtkMRMLScalarVolumeNode"):
                        print("Can't find volumeNode")
                        continue  # Preskoči, če ni veljaven volume

                    # Preveri, če volume ima StorageNode (drugače `.GetFileName()` vrže napako)
                    storageNode = volumeNode.GetStorageNode()
                    
                    
                    if not storageNode:
                        print("Can't find storageNode")
                        continue  # Preskoči, če volume nima shranjenih DICOM podatkov
                    volumeName = volumeNode.GetName()
                    #print(volumeName)
                    imageItem = shNode.GetItemByDataNode(volumeNode)
                    #print(imageItem)
                    #dicomUIDsList = volumeNode.GetAttribute("DICOM.instanceUIDs").split()
                    # Preverimo modaliteto volumna (DICOM metapodatki)
                    #modality = slicer.dicomDatabase.fileValue(storageNode.GetFileName(), "0008,0060")              #prazen   
                    #modality = volumeNode.GetAttribute("DICOM.Modality")                                           #None
                    #modality = slicer.dicomDatabase.fileValue(uid, "0008,0060")  # Modality                        #prazen
                    #modality = slicer.dicomDatabase.fileValue(dicomUIDsList[0], "0008,0060")                       #prazen
                    modality = shNode.GetItemAttribute(imageItem, "DICOM.Modality")                                 #deluje!
                    #print(modality)

                    dimensions = volumeNode.GetImageData().GetDimensions()
                    spacing = volumeNode.GetSpacing()
                    #print(f"Volume {volumeNode.GetName()} - Dimenzije: {dimensions}, Spacing: {spacing}")

                    if modality != "CT":
                        print("Not a CT")
                        continue  # Preskoči, če ni CT

                    
                    # Preveri, če volume obstaja v sceni
                    if not slicer.mrmlScene.IsNodePresent(volumeNode):
                        print(f"Volume {volumeName} not present in the scene.")
                        continue

                    # Preverimo proizvajalca (DICOM metapodatki)
                    manufacturer = shNode.GetItemAttribute(imageItem, 'DICOM.Manufacturer')
                    #manufacturer = volumeNode.GetAttribute("DICOM.Manufacturer")
                    #manufacturer = slicer.dicomDatabase.fileValue(uid, "0008,0070")
                    #print(manufacturer)
                    # Določimo, ali gre za CBCT ali CT
                    if "varian" in manufacturer.lower() or "elekta" in manufacturer.lower():
                        cbct_list.append(volumeName)
                        scan_type = "CBCT"
                        yesCbct = True
                        print("CBCT")
                    else:  # Siemens ali Philips
                        ct_list.append(volumeName)
                        scan_type = "CT"
                        yesCbct = False
                        print("CT")

                    # Detekcija točk v volumnu
                    grouped_points = detect_points_region_growing(volumeName, yesCbct, intensity_threshold=3000)
                    #print(f"Populating volume_points_dict with key ('{scan_type}', '{volumeName}')")
                    volume_points_dict[(scan_type, volumeName)] = grouped_points
                    #print(volume_points_dict)  # Check if the key is correctly added

            # Če imamo oba tipa volumna (CBCT in CT) **znotraj istega studyja**
            if cbct_list and ct_list:
                ct_volume_name = ct_list[0]  # Uporabi prvi CT kot referenco
                ct_points = [centroid for centroid, _ in volume_points_dict[("CT", ct_volume_name)]]

                if len(ct_points) < 3:
                    print(f"CT volume {ct_volume_name} doesn't have enough points for registration.")
                else:
                    for cbct_volume_name in cbct_list:
                        cbct_points = [centroid for centroid, _ in volume_points_dict[("CBCT", cbct_volume_name)]]

                        print(f"\nProcessing CBCT Volume: {cbct_volume_name}")
                        if len(cbct_points) < 3:
                            print(f"CBCT Volume '{cbct_volume_name}' doesn't have enough points for registration.")
                            continue

                        # Shranjevanje razdalj
                        distances_ct_cbct = []
                        distances_internal = {"A-B": [], "B-C": [], "C-A": []}

                        cbct_points_array = np.array(cbct_points)  # Pretvorba v numpy array

                        ct_volume_node = slicer.util.getNode(ct_volume_name)
                        cbct_volume_node = slicer.util.getNode(cbct_volume_name)
                        ct_spacing = ct_volume_node.GetSpacing()  # (x_spacing, y_spacing, z_spacing)
                        cbct_spacing = cbct_volume_node.GetSpacing()  # (x_spacing, y_spacing, z_spacing)
                        
                        ct_scale_factor = np.array(ct_spacing)  # Spacing za CT (x, y, z)                        
                        cbct_scale_factor = np.array(cbct_spacing)  # Spacing za CBCT (x, y, z)
                        print(ct_scale_factor, cbct_scale_factor)

                        
                        # Sortiramo točke po Z-koordinati (ali X/Y, če raje uporabljaš drugo os)
                        cbct_points_sorted = cbct_points_array[np.argsort(cbct_points_array[:, 2])]

                        # Razdalje med CT in CBCT (SORTIRANE točke!)
                        d_ct_cbct = np.linalg.norm(cbct_points_sorted - ct_points, axis=1)
                        distances_ct_cbct.append(d_ct_cbct)

                        # Razdalje med točkami znotraj SORTIRANIH cbct_points
                        d_ab = np.linalg.norm(cbct_points_sorted[0] - cbct_points_sorted[1])
                        d_bc = np.linalg.norm(cbct_points_sorted[1] - cbct_points_sorted[2])
                        d_ca = np.linalg.norm(cbct_points_sorted[2] - cbct_points_sorted[0])

                        # Sortiramo razdalje po velikosti, da so vedno v enakem vrstnem redu
                        sorted_distances = sorted([d_ab, d_bc, d_ca])

                        distances_internal["A-B"].append(sorted_distances[0])
                        distances_internal["B-C"].append(sorted_distances[1])
                        distances_internal["C-A"].append(sorted_distances[2])
                        
                        # Dodamo ime študije za v statistiko
                        studyName = shNode.GetItemName(studyItem)
                        
                        # **Shrani razdalje v globalne sezname**
                        prostate_size_est.append({"Study": studyName, "Distances": sorted_distances})
                        ctcbct_distance.append({"Study": studyName, "Distances": list(distances_ct_cbct[-1])})  # Pretvorimo v seznam

                        # Izberi metodo glede na uporabnikov izbor
                        chosen_rotation_matrix = np.eye(3)
                        chosen_translation_vector = np.zeros(3)

                        if applyScaling:
                            scaling_factors = compute_optimal_scaling_per_axis(cbct_points, ct_points)
                            print("Scaling factors: ", scaling_factors)
                            cbct_points = compute_scaling(cbct_points, scaling_factors)

                        if applyRotation:
                            if selectedMethod == "Kabsch":
                                chosen_rotation_matrix = compute_Kabsch_rotation(cbct_points, ct_points)
                            elif selectedMethod == "Horn":
                                chosen_rotation_matrix = compute_Horn_rotation(cbct_points, ct_points)
                            elif selectedMethod == "Iterative Closest Point (Kabsch)":
                                _, chosen_rotation_matrix, _ = icp_algorithm(cbct_points, ct_points)
                            print("Rotation Matrix:\n", chosen_rotation_matrix)

                        if applyTranslation:
                            chosen_translation_vector = compute_translation(cbct_points, ct_points, chosen_rotation_matrix)
                            print("Translation Vector:\n", chosen_translation_vector)

                        # Ustvari vtkTransformNode in ga poveži z CBCT volumenom
                        imeTransformNoda = cbct_volume_name + " Transform"
                        transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", imeTransformNoda)

                        # Kreiraj transformacijo in jo uporabi
                        vtk_transform = create_vtk_transform(chosen_rotation_matrix, chosen_translation_vector)
                        transform_node.SetAndObserveTransformToParent(vtk_transform)

                        # Pridobi CBCT volumen in aplikacijo transformacije
                        cbct_volume_node = slicer.util.getNode(cbct_volume_name)
                        cbct_volume_node.SetAndObserveTransformNodeID(transform_node.GetID())

                        # Uporabi transformacijo na volumnu (fizična aplikacija)
                        slicer.vtkSlicerTransformLogic().hardenTransform(cbct_volume_node) #aplicira transformacijo na volumnu
                        slicer.mrmlScene.RemoveNode(transform_node) # Odstrani transformacijo iz scene
                        print("Transform successful on ", cbct_volume_name)

            else:
                print(f"Study {studyItem} doesn't have any appropriate CT or CBCT volumes.")

        # Izpis globalne statistike
        
        
        if(writefilecheck):
            print("Distances between CT & CBCT markers: ", ctcbct_distance)
            print("Distances between pairs of markers for each volume: ", prostate_size_est)
            # Define the file path for the CSV file
            file_path = os.path.join(os.path.dirname(__file__), "study_data.csv")

            # Write lists to the CSV file
            with open(file_path, mode='w', newline='') as file: #w za write, a za append
                writer = csv.writer(file)
                # Write headers
                writer.writerow(["Prostate Size", "CT-CBCT Distance"])
                # Write data rows
                for i in range(len(prostate_size_est)):
                    writer.writerow([prostate_size_est[i], ctcbct_distance[i]])
                print("File written at ", file_path)

