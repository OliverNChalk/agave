# Invalidator Client Scripts
Scripts for interacting with the Solana invalidator client for test automation purposes.

# File Overviews

## continuous-test.sh
the main entry point that parses params and kicks off appropriate downstream testplan.

## invalidator-tests.sh
Contains the individual attack definitions.

## testplan/
Collections of attacks to run.

### private-testnet.sh
Essentially all of the attacks that have been developed.

### private-testnet-no-accounts.sh
All of the attacks exccept the ones that require prior account/program state setup.

### public-testnet.sh
All of the attacks not expected to bring down testnet.
