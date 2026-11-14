"""
Unit tests for MockCommunicator and collective communication operations.
"""

import unittest

from dmg.tensor_parallel.communicator import (
    MockCommunicator,
    MockCommunicatorGroup,
    create_communicator,
)
from dmg.tensor_parallel.config import CommBackend, TensorParallelConfig


class TestCommunicatorCollectives(unittest.TestCase):
    """Test suite for distributed collective operations."""

    def setUp(self):
        self.world_size = 4
        self.group = MockCommunicatorGroup(world_size=self.world_size)
        self.communicators = [
            MockCommunicator(
                TensorParallelConfig(
                    world_size=self.world_size, rank=r, backend=CommBackend.MOCK
                ),
                shared_group=self.group,
            )
            for r in range(self.world_size)
        ]

    def test_all_reduce_sum_lists(self):
        # Each rank provides a 2x2 matrix
        inputs = [
            [[1 * (10**r), 2 * (10**r)], [3 * (10**r), 4 * (10**r)]]
            for r in range(self.world_size)
        ]

        # All ranks submit their tensor for operation 'reduce_test'
        for r in range(self.world_size):
            self.communicators[r].submit_tensor("reduce_test", inputs[r])

        # All ranks retrieve reduced sum
        results = [
            self.communicators[r].all_reduce_sum(op_id="reduce_test")
            for r in range(self.world_size)
        ]

        expected = [[1111, 2222], [3333, 4444]]
        for res in results:
            self.assertEqual(res, expected)

    def test_all_gather(self):
        # Each rank provides 1 row of 3 elements
        inputs = [[[r, r * 2, r * 3]] for r in range(self.world_size)]

        for r in range(self.world_size):
            self.communicators[r].submit_tensor("gather_test", inputs[r])

        results = [
            self.communicators[r].all_gather(op_id="gather_test", dim=0)
            for r in range(self.world_size)
        ]

        expected = [
            [0, 0, 0],
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9],
        ]
        for res in results:
            self.assertEqual(res, expected)

    def test_broadcast(self):
        src_payload = [[42, 99], [7, 13]]
        # Rank 0 submits payload
        self.communicators[0].submit_tensor("bcast_test", src_payload)

        # All ranks receive broadcast from rank 0
        for r in range(self.world_size):
            res = self.communicators[r].broadcast(src_rank=0, op_id="bcast_test")
            self.assertEqual(res, src_payload)

    def test_factory_creation(self):
        cfg = TensorParallelConfig(world_size=2, rank=0, backend=CommBackend.MOCK)
        comm = create_communicator(cfg)
        self.assertIsInstance(comm, MockCommunicator)
        self.assertEqual(comm.rank, 0)
        self.assertEqual(comm.world_size, 2)


if __name__ == "__main__":
    unittest.main()
