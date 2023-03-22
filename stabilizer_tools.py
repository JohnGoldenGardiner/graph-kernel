import warnings
import numpy as np
from collections import defaultdict

class Stabilizer:
    """
    Signed Pauli string
    """

    def __init__(self, num_qubits, pauli_z=None):
        """
        Args:
            num_qubits: number of qubits on which stabilizer acts
            pauli_z: if this is an integer, the stabilizer II...IZI...I is 
                returned where the Z acts on the qubit designated by the 
                integer
        
        """
        self.z = np.zeros(num_qubits, dtype=int)
        if pauli_z is not None:
            self.z[pauli_z] = 1
        self.x = np.zeros(num_qubits, dtype=int)
        self.sign = 0

    def __matmul__(self, other):
        
        if not isinstance(other, Stabilizer):
            raise TypeError('Stabilizer objects can only be matrix multiplied '
                            'by other Stabilizer objects')
        
        if len(other.z) != len(self.z):
            raise ValueError('The stabilizers must act on the same number '
                             'of qubits')

        product = Stabilizer(len(self.z))
        product.z = (self.z + other.z) % 2
        product.x = (self.x + other.x) % 2
        commutation_factor = 2*np.sum(self.x*other.z)
        product.sign = (self.sign + other.sign + commutation_factor) % 4

        return product
    
    def __mul__(self, other):

        if isinstance(other, Stabilizer):
            raise TypeError('Stabilizer objects can only be multiplied by '
                            'scalars. Use matrix multiplication @ instead')

        if not np.isscalar(other):
            raise TypeError('Stabilizer objects can only be multiplied by '
                            'scalars')

        possible_coefficients = [1, 1j, -1, -1j]
        if other not in possible_coefficients:
            raise TypeError('Stabilizer objects can only be multiplied by '
                            '1, 1j, -1, or -1j')

        product = Stabilizer(len(self.z))
        product.z = self.z
        product.x = self.x
        product.sign = self.sign + possible_coefficients.index(other)

        return product
    
    def __rmul__(self, other):
        return self*other

    def __eq__(self, other):

        if not isinstance(other, Stabilizer):
            return False
        
        is_equal = (np.all(self.z == other.z) 
                    and np.all(self.x == other.x) 
                    and np.all(self.sign == other.sign))
        return is_equal
    
    def copy(self):
        num_qubits = len(self.z)
        new = Stabilizer(num_qubits)
        new.z = self.z.copy()
        new.x = self.x.copy()
        new.sign = self.sign
        return new

    def equal_up_to_sign(self, other):
        return np.all(self.z == other.z) and np.all(self.x == other.x)

    def commute_test(self, other):
        if not isinstance(other, Stabilizer):
            raise TypeError('Argument must be Stabilizer object')
        
        if len(other.z) != len(self.z):
            raise ValueError('Argument must act on the same number of qubits '
                             'as self')
        
        if np.sum(self.z*other.x + self.x*other.z) % 2 == 0:
            return True
        else:
            return False


class StabilizerTableau:
    """
    Tableau of signed Pauli strings
    """

    def __init__(self, stabilizers, destabilizers=False):
        """
        Args:
            stabilizers: either a list of Stabilizer objects or a positive 
                integer. If an integer n then it is initialized to the stabilizer 
                tableau representing the |0>^n state. I.e. if stabilizers is 
                an integer n and destabilizers=False, the Z content is the n 
                by n identity array and the X content is the n by n zero array. 
                If stabilizers is an integer and destabilizers=True, the Z content 
                is the zero array and the X content is the identity array.

            destabilizers: determines whether the Z or X content is the n by n
                identity when stabilizers is an integer n. Set to True to 
                initialize a destabilizer tableau for the |0>^n state.
        """
            
        if isinstance(stabilizers, int):
            num_qubits = stabilizers
            if destabilizers == False:
                self.z = np.eye(num_qubits, dtype=int)
                self.x = np.zeros((num_qubits, num_qubits), dtype=int)
            else:
                self.x = np.eye(num_qubits, dtype=int)
                self.z = np.zeros((num_qubits, num_qubits), dtype=int)
            self.signs = np.zeros(num_qubits, dtype=int)

        else:
            num_stabilizers = len(stabilizers)
            num_qubits = len(stabilizers[0].z)
            self.z = np.zeros((num_qubits, num_stabilizers), dtype=int)
            self.x = np.zeros((num_qubits, num_stabilizers), dtype=int)
            self.signs = np.ones(num_stabilizers, dtype=int)
            for i, stab in enumerate(stabilizers):
                self.z[:, i] = stab.z
                self.x[:, i] = stab.x
                self.signs[i] = stab.sign

    def __getitem__(self, index):
        """
        Return a chosen stabilizer from the tableau of stabilizers

        Args:
            index: integer specifying the stabilizer to return

        Returns:
            the index-th stabilizer

        Note:
            This does not return a "view" of the StabilizerTableau object
            but rather instantiates a new Stabilizer object and fills it 
            with copies.
        """
        num_qubits = self.z.shape[0]
        stabilizer = Stabilizer(num_qubits)
        stabilizer.z = self.z[:, index].copy()
        stabilizer.x = self.x[:, index].copy()
        stabilizer.sign = self.signs[index]
        return stabilizer
    
    def __setitem__(self, index, stabilizer: Stabilizer):
        """
        Set a chosen stabilizer. Change self in place to have stabilizer
        as the index-th stabilizer in the tableau.

        Note:
            This copies the stabilizer argument.
        """
        self.z[:, index] = stabilizer.z.copy()
        self.x[:, index] = stabilizer.x.copy()
        self.signs[index] = stabilizer.sign

    def __str__(self):
        num_qubits = self.z.shape[0]
        s = ''
        for i in range(num_qubits):
            s += str(self.z[i]) + '\n'
        s += '-'*len(str(self.z[0])) + '\n'
        for i in range(num_qubits):
            s += str(self.x[i]) + '\n'
        s += '-'*len(str(self.z[0])) + '\n'
        s += str(self.signs)
        return s

    def copy(self):
        new = StabilizerTableau(1)
        new.z = self.z.copy()
        new.x = self.x.copy()
        new.signs = self.signs.copy()
        return new
    
    def destabilizer_check(self, other):
        """
        Check that other is a valid destabilizer tableau for self

        Args:
            self: StabilizerTableau representing a stabilizer tableau
            other: StabilizerTableau that may or may not be a destabilizer 
                tableau for self
        
        Returns:
            True if destabilizer tableau describes a valid set of 
            destabilizers for self in the correct order
        """
        if not isinstance(other, StabilizerTableau):
            raise TypeError('other must be a StabilizerTableau')
        
        if self.z.shape != other.z.shape:
            raise ValueError('other must have same size as self')
        
        num_stabilizers = self.z.shape[1]
        for i, j in np.ndindex((num_stabilizers, num_stabilizers)):
            if not (i == j) ^ other[i].commute_test(self[j]):
                return False
        else:
            return True
    
    def conjugate(self, gate, *qubits):
        """
        Conjugate stabilizers by a specified gate (in place)
        """

        # Parse *qubits arguments
        if len(qubits) == 1:
            q = qubits[0]
            if not isinstance(q, int):
                raise TypeError('Qubits must be int type')
        elif len(qubits) == 2:
            source = qubits[0]
            target = qubits[1]
            if not isinstance(source, int) or not isinstance(target, int):
                raise TypeError('Qubits must be int type')
        else:
            raise ValueError(f'{len(qubits)} qubits given when 1 or 2 expected')

        # Act with gate
        if gate == 'h':
            q = qubits[0]
            self.signs = (self.signs + 2*self.z[q]*self.x[q]) % 4
            self.z[q], self.x[q] = (self.x[q].copy(), self.z[q].copy())
        
        elif gate == 's':
            q = qubits[0]
            self.signs = (self.signs - self.x[q]) % 4
            self.z[q] = (self.z[q] + self.x[q]) % 2

        elif gate == 'z':
            q = qubits[0]
            self.signs = (self.signs - 2*self.x[q]) % 4
        
        elif gate == 'sdg':
            q = qubits[0]
            self.signs = (self.signs + self.x[q]) % 4
            self.z[q] = (self.z[q] + self.x[q]) % 2

        elif gate == 'sx':
            q = qubits[0]
            self.signs = (self.signs + self.z[q]) % 4
            self.x[q] = (self.x[q] + self.z[q]) % 2

        elif gate == 'x':
            q = qubits[0]
            self.signs = (self.signs + 2*self.z[q]) % 4

        elif gate == 'sxdg':
            q = qubits[0]
            self.signs = (self.signs - self.z[q]) % 4
            self.x[q] = (self.x[q] + self.z[q]) % 2

        elif gate == 'sy':
            q = qubits[0]
            self.signs = (self.signs + 2*self.x[q]*(1 - self.z[q])) % 4
            self.z[q], self.x[q] = (self.x[q].copy(), self.z[q].copy())

        elif gate == 'y':
            q = qubits[0]
            self.signs = (self.signs + 2*self.z[q] + 2*self.x[q]) % 4

        elif gate == 'sydg':
            q = qubits[0]
            self.signs = (self.signs + 2*self.z[q]*(1 - self.x[q])) % 4
            self.z[q], self.x[q] = (self.x[q].copy(), self.z[q].copy())

        elif gate == 'cz':
            source, target = qubits
            self.signs = (self.signs + 2*self.x[source]*self.x[target]) % 4
            self.z[target] = (self.z[target] + self.x[source]) % 2
            self.z[source] = (self.z[source] + self.x[target]) % 2

        elif gate == 'cx':
            source, target = qubits
            self.z[source] = (self.z[source] + self.z[target]) % 2
            self.x[target] = (self.x[target] + self.x[source]) % 2

        elif gate == 'id':
            pass

        else:
            raise ValueError(f'The gate {gate} is not implemented')

    def measure_pauli_string(
            self, 
            pauli_string: Stabilizer, 
            destabilizers=None
        ) -> int:
        """
        Measure a signed Pauli string observable
        
        Args:
            pauli_string: a Pauli string observable to be measured

        Returns:
            measurement result of 1 or -1,

        Notes:
            This makes an in-place change to the StabilizerTableau object (as 
            well as the destabilizer argument if given)! The stabilizer 
            tableau (and destabilizer tableau) is updated to represent the 
            post-measurement state.
        """
        num_qubits = self.z.shape[0]
        num_stabilizers = self.z.shape[1]

        if destabilizers is None:
            warnings.warn('Running measure_pauli_string() without providing '
                          'destabilizers is slow. Consider obtaining '
                          'destabilizers for your stabilizer tableau', 
                          RuntimeWarning)

            # Find non-commuting stabilizers
            for i in range(num_stabilizers):
                if not pauli_string.commute_test(self[i]):
                    i0 = i
                    break
            else:
                # Perform Gaussian elimination

                # Build augmented matrix
                mat_z = np.column_stack((self.z, pauli_string.z))
                mat_x = np.column_stack((self.x, pauli_string.x))
                mat = np.concatenate((mat_z, mat_x), axis=0)

                # Make row echelon form
                found_pivot = False
                pivots = []
                pivot_row = 0
                for column in range(num_stabilizers):
                    for row in range(pivot_row, 2*num_qubits):
                        if mat[row, column] == 1:
                            found_pivot = True
                            pivots.append(column)
                            mat[[row, pivot_row]] = mat[[pivot_row, row]]
                            break
                    row0 = row
                    for row in range(row0 + 1, 2*num_qubits):
                        if mat[row, column] == 1:
                            mat[row] = (mat[row] - mat[pivot_row]) % 2
                    if found_pivot:
                        pivot_row += 1

                # Reduce pivot columns
                for row0, pivot in reversed(list(enumerate(pivots))):
                # for row0, pivot in zip(range(num_pivots, -1, -1), pivots[::-1]):
                    for row in range(row0):
                        if mat[row, pivot] == 1:
                            mat[row] = (mat[row] - mat[row0]) % 2

                # Read off result
                result = np.zeros(num_stabilizers, dtype=int)
                for i, pivot in enumerate(pivots):
                    result[pivot] = mat[i, -1]

                # Check result
                product = Stabilizer(num_qubits)
                for i, present in enumerate(result):
                    if present == 1:
                        product = self[i] @ product
                assert pauli_string.equal_up_to_sign(product), ('Gaussian '
                                                                'elimination '
                                                                'failed')

                # Extract sign difference
                sign = (pauli_string.sign - product.sign) % 4
                assert sign == 0 or sign == 2, ('Imaginary measurement result. '
                                                'Check that pauli_string is '
                                                'Hermitian')
                eigenvalue = (-1)**(sign//2)

                return eigenvalue

            # Random measurement outcome
            random_sign = (-1)**np.random.randint(2)
            # Update state
            for i in range(i0 + 1, num_stabilizers):
                if not pauli_string.commute_test(self[i]):
                    self[i] = self[i0] @ self[i]
            self[i0] = random_sign*pauli_string

            return random_sign

        if not isinstance(destabilizers, StabilizerTableau):
            raise TypeError('destabilizers must be a StabilizerTableau object')
        if destabilizers.z.shape != (num_qubits, num_stabilizers):
            raise ValueError('destabilizer tableau must be the same size')

        # TODO: Test this routine
        for j in range(num_stabilizers):
            if not pauli_string.commute_test(self[j]):
                j0 = j
                break
        else:
            # Get sign
            product = Stabilizer(num_qubits)
            for i in range(num_stabilizers):
                if not pauli_string.commute_test(destabilizers[i]):
                    product = self[i] @ product
            sign = (product.sign - pauli_string.sign) % 4

            assert pauli_string.equal_up_to_sign(product), ('Gaussian '
                                                            'elimination '
                                                            'failed')
            assert sign == 0 or sign == 2, ('Imaginary measurement result. '
                                            'Check that pauli_string is '
                                            'Hermitian')
            
            # Result
            eigenvalue = (-1)**(sign//2)
            return eigenvalue

        random_sign = (-1)**np.random.randint(2)
        for j in range(j0 + 1, num_stabilizers):
            if not pauli_string.commute_test(self[j]):
                self[j] = self[j0] @ self[j]
        for j in range(num_stabilizers):
            if not pauli_string.commute_test(self[j]):
                destabilizers[j] = self[j0] @ destabilizers[j]
        
        destabilizers[j0] = self[j0]
        self[j0] = random_sign*pauli_string

        return random_sign

    def measure_qubit(self, qubit, destabilizers=None):
        """
        Measure a qubit in the Z basis
        
        Args:
            qubit: the qubit to be measured

        Returns:
            measurement result of 1 or -1,

        Notes:
            This makes an in-place change to the StabilizerTableau object! 
            The stabilizer tableau is updated to represent the post-measurement
            state.
        """

        num_qubits = self.z.shape[0]
        num_stabilizers = self.z.shape[1]

        if destabilizers is None:
            pauli_string = Stabilizer(num_qubits)
            pauli_string.z[qubit] = 1
            return self.measure_pauli_string(pauli_string)

        if not isinstance(destabilizers, StabilizerTableau):
            raise TypeError('destabilizers must be a StabilizerTableau object')

        for j in range(num_stabilizers):
            if self.x[qubit, j]:
                j0 = j
                break
        else:
            # Get sign
            product = Stabilizer(num_qubits)
            for i, present in enumerate(destabilizers.x[qubit]):
                if present == 1:
                    product = self[i] @ product
            sign = product.sign

            # # Assertions
            # assert not np.any(product.x), ('Product of destabilizers '
            #                                 'is incorrect')
            # assert all([
            #     product.z[i] == 1 if i == qubit 
            #     else product.z[i] == 0 
            #     for i in range(num_qubits)
            # ]), 'Product of destabilizers is incorrect'
            # assert sign == 0 or sign == 2, ('Imaginary measurement result. '
            #                                 'Check that pauli_string is '
            #                                 'Hermitian')
            
            # Result
            eigenvalue = (-1)**(sign//2)
            return eigenvalue
        
        for j in range(j0 + 1, num_stabilizers):
            if self.x[qubit, j]:
                self[j] = self[j0] @ self[j]
        for j in range(num_stabilizers):
            if destabilizers.x[qubit, j]:
                destabilizers[j] = self[j0] @ destabilizers[j]
        destabilizers[j0] = self[j0]
        random_sign = (-1)**np.random.randint(2)
        self[j0] = random_sign*Stabilizer(num_qubits, pauli_z=qubit)

        return random_sign

    def sample_z_basis(self, destabilizers=None, shots=1024):
        """
        Record results of measurements in the Z-basis

        Args:
            shots: Number of times to repeat the measurement on the original 
                state

        Returns:
            a dictionary with counts for each encountered outcome

        Note:
            This does not change the StabilizerTableau, unlike 
            StabilizerTableau.measure_pauli_string()!
        """

        num_qubits = self.z.shape[0]
        if (
            destabilizers is not None 
            and not self.destabilizer_check(destabilizers)
            ):
            raise ValueError('Given destabilizers are not in fact '
                             'destabilizers')

        results = defaultdict(int)
        for shot in range(shots):
            
            stabs = self.copy()
            destabs = destabilizers.copy()
            num_qubits = stabs.z.shape[0]
            
            result = ''
            for qubit in range(num_qubits):
                eigenvalue = stabs.measure_qubit(qubit, destabilizers=destabs)

                assert eigenvalue == 1 or eigenvalue == -1, 'Wrong outcome'
                if eigenvalue == 1: qubit_result = '0'
                else: qubit_result = '1'
                
                result += qubit_result
        
            results[result] += 1

        return results