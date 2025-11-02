#include <stdio.h>
#include <stdlib.h>
#include "knapsack_c.h"

int main() {
  const char *cfg = "{\"mode\":\"select\",\"items\":[],\"objective\":[0],\"constraints\":[{\"capacity\":0,\"weights\":[0]}]}";
  const char *opts = NULL;
  struct KnapsackSolutionV2 *out = NULL;
  int rc = solve_knapsack_v2_from_json(cfg, opts, &out);
  if (rc != 0) {
    printf("solve returned %d (expected nonzero on empty)\n", rc);
  }
  if (out) free_knapsack_solution_v2(out);
  return 0;
}
