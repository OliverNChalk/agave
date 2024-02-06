# Synchronizing changes with the upstream

**NOTE** This document is targeted at someone actually merging changes from the
upstream.  If you are just using this repo for development, you do not
necessarily need to read it or to understand all the details.

We need to merge changes from the upstream repository into this repository
regularly, in order to reduce merge conflicts and to stay up to date.

For now, we are going to track only the `master` branch of the upstream
repository.  That would be `master` in `https://github.com/solana-labs/solana`.

`invalidator` will use the following branches to track state relative to the
upstream Solana repo.  Note that all branches with the `sync/` prefix are
created and/or updated only during the synchronization process.

 * `sync/master/upstream/[date]`<br/>
   Records state of the `master` branch in the upstream as of the specified
   date.  For example `sync/master/upstream/2023-04-03` would be one such
   branch.<br/>
   These are created for every synchronization.  And we probably will keep them
   around for at least a few weeks after.<br/>

 * `sync/master/local/[date]`<br/>
   Records state of the `master` branch before the synchronization on the
   specified date.<br/>
   These branches help rebase local changes and PRs, after `master` is
   rebased.<br/>

 * `sync/master-upstream`<br/>
   Points to the latest `sync/master/upstream/[date]` branch that is
   currently used as a basis for the `invalidator` `master` branch.

 * `sync/master-local`<br/>
   Points to the latest `sync/master/local/[date]` branch.  It is holding
   any changes local to `invalidator` that have not been merged upstream
   (`sync/master-upstream`) that were present at the point of the last
   synchronization.

 * `master`<br/>
   Contains all the changes in `sync/master-upstream`, plus all changes from
   `sync/master-local`, plus all changes in the `invalidator` that were created
   since the last synchronization.

One additional local branch is used during the synchronization process:

 * `master-next`<br/>
   Contains changes from `master`, rebased on top of an updated
   `sync/master-upstream`.

## Synchronization process

Synchronization with the upstream happens in a few steps.

In order to avoid accidentally pushing `invalidator` changes into the public
repository it is strongly recommended that you use a dedicated workspace for
synchronization.  With remote names that would reduce this kind of push.  You
can create this setup as follows:

```sh
git clone --origin invalidator git@github.com:solana-labs/invalidator.git
cd invalidator
git remote add upstream https://github.com/solana-labs/solana.git
git remote set-url --push upstream do-not-push-from-invalidator-to-solana
```

Commands below assume `invalidator` and `upstream` remotes, setup as specified
above.

```text
$ git remote -v
invalidator     git@github.com:solana-labs/invalidator.git (fetch)
invalidator     git@github.com:solana-labs/invalidator.git (push)
upstream        https://github.com/solana-labs/solana.git (fetch)
upstream        do-not-push-from-invalidator-to-solana (push)
```

### 1.1. Record update date.

The following steps are run by a single person.  Preferably on Monday.

All commands below assume you recorded the current date as:

```sh
export SYNC_DATE=$(TZ=UTC date "+%Y-%m-%d")
```

### 1.2. Process automation.

Considering the complexity of the synchronization process it probably makes
sense to write a script that would execute all the steps, rather than asking
people to copy/paste them from this document.

It makes sense to do it a bit later, after we run a number of sync operations
and will see that the process does not change any more.  When most of the tricky
cases are addressed and recorded in this document.

This process was tried in the `solana-replay-invalidator` repo and went through
roughly 3 versions before being adapted for the `invalidator` repo.

### 1.3. Lock `master`

As a branch update is a potentially destructive operation it is important to
make sure that nobody will add any new commits to the `master` branch while the
synchronization process is being executed.

For now, to avoid any loss of work, I suggest `master` is set as a protected
branch in GitHub for the duration of the sync, starting here.  Make sure to add
yourself to the exclusion list, so you can update `master` even while it is
protected, but nobody else can.

***TODO*** Describe exact steps for this.

***TODO*** It is possible to reduce the time `master` is blocked to just a short
span of time when an updated `master-next` is been pushed into `master`, rather
than for the whole duration of the synchronization process.

### 1.4. Create a new sync point, and `master-next`.

We want to use a commit from `solana/master` that is the last change before
00:00 UTC on Monday.  Hopefully that would be a rather low activity point.

Fetch the latest state of the `https://github.com/solana-labs/solana` `master`,
and point `sync/master/upstream/$SYNC_DATE` at the commit just before Monday
starts in UTC:

```sh
git fetch upstream master
git branch --no-track "sync/master/upstream/$SYNC_DATE" \
    "$( TZ=UTC \
        git log --max-count=1 \
            --until="$SYNC_DATE 00:00:00+00:00" \
            --format=format:%H \
            upstream/master \
    )"

git branch --no-track sync/master-upstream "sync/master/upstream/$SYNC_DATE"

git fetch invalidator master
git branch --no-track "sync/master/local/$SYNC_DATE" invalidator/master
git branch --no-track "sync/master-local" "sync/master/local/$SYNC_DATE"

git switch --create master-next --no-track sync/master-local
```

### 1.5. Run `cargo check` and `cargo fmt` for `invalidator`

While the `invalidator` repo does not have a working CI, it might be desirable
to run `cargo check` and `cargo fmt` on each commit in the rebased branch
beforehand.  To make sure existing issues are not mixed with the new ones:

```sh
git rebase --interactive --no-autosquash \
    --reschedule-failed-exec \
    --exec "./cargo check --lib --bins --tests" \
    --exec "./scripts/cargo-fmt.sh" \
    --exec "cd programs/sbf && \
        ../../cargo check --bins --tests" \
    "$( git merge-base master-next sync/master-upstream )"
```

This will run an interactive rebase, making sure that every commit compiles and
is correctly formatted, according to the currently used rustfmt.

If any discrepancy is found by the `exec` command, and it is a small enough fix,
such as invalid formatting, fix it and then apply changes to the last commit in
the new `master-next` history:

```sh
git add --update
git commit --amend --reuse-message
```

The very last commit should be the one that caused a failure, as you must have
checked all the commits before already.

Check that the issue is resolved, and the continue checking:

```sh
./cargo check --lib --bins --tests && ./scripts/cargo-fmt.sh
git rebase --continue
```

Until the whole `master-next` branch is checked.

### 1.6. Rebase `master-next` on top of the new sync point

Rebase changes on top of the updated upstream master.

As there are generally enough changes accumulated in a week, it is better to
rebase one upstream change at a time, compiling after every rebase.  This will
identify conflicts as soon as they arise, making it easier to resolve them.

In the workspace root, run:

```sh
./scripts/slow-rebase.sh --date "$SYNC_DATE"
```

The script will run the rebase, one upstream commit at a time, making sure that
the codebase compiles and is correctly formatted.  If anything fails the script
should tell you what to do next and how to restart the rebase process.

Ideally, we would compile and format every commit, rather than doing it only
once after everything rebased on top of the next upstream commit.  This way, the
compilation or formatting is immediately attributed to the right commit in the
`master-next` branch.  But, is too time consuming.

If you created any "DO NOT SUBMIT: Fixup" commits per script instructions, after
a complete rebase is done, I do another run like this:

```sh
git rebase --interactive \
    --reschedule-failed-exec \
    --exec "./cargo check --lib --bins --tests" \
    --exec "./scripts/cargo-fmt.sh" \
    --exec "./cargo nightly clippy --workspace --all-targets --features dummy-for-ci-check -- \
            --deny=warnings \
            --deny=clippy::default_trait_access \
            --deny=clippy::arithmetic_side_effects \
            --deny=clippy::manual_let_else \
            --deny=clippy::used_underscore_binding \
            --allow=clippy::redundant_clone" \
    --exec "cd programs/sbf \
        && ../../cargo check --bins --tests" \
    "$( git merge-base sync/master-upstream HEAD )"
```

It redistributes the fixups into the right commits, possibly with minor
adjustments.

If you need to see relative state of the branches, run:

```sh
git log --graph --decorate --oneline \
    sync/master-upstream sync/master-upstream \
    master master-next HEAD \
    --not sync/master-upstream^
```

Or, if you want to see the full picture:

```sh
git log --graph --decorate --oneline \
    --branches='invalidator/sync/master/upstream/*' \
    --branches='invalidator/sync/master/local/*' \
    invalidator/sync/master-upstream invalidator/sync/master-local \
    sync/master-upstream sync/master-local \
    upstream/master \
    master-next HEAD
```

If you are trying to resolve a conflict and want to see a change in the upstream
that caused the conflict run

```sh
git show $( git rev-list master-next..sync/master-upstream | tail -n 1 )
```

If you want to be even more careful, you could run `cargo test` after every
merge, though, it could be too slow, if the `invalidator` changes reach 100+
commits:

```sh
./scripts/slow-rebase.sh --date "$SYNC_DATE" --run-tests
```

#### 1.6.1. Frozen ABI hashes

If a change affects a type with an `AbiExample` instance, it may change the hash
value for this type.  A few types in the `invalidator` repo have done this.  In
particular, change that extracted `solana-net-protocol` caused at least one hash
to change.

You can wait for the CI to tell you if any of the hashes mismatch.

Alternatively, you can check it yourself either by running

```sh
./test-abi.sh
```

to check the whole repo.  Or you can run `cargo test` with the nightly compiler,
explicitly specifying the package, if you want to save some:

```sh
./cargo nightly test --package solana-net-protocol --lib -- test_abi_
```

### 1.7. Run CI for `master-next`

Create a PR with the `master-next` content in the `invalidator` repo and wait
for a successful CI result.

```sh
git push origin --force master-next
```

### 1.8. Publish updated `master` and `sync/*` branches

```sh
git push invalidator "sync/master/upstream/$SYNC_DATE"
git push invalidator "sync/master/local/$SYNC_DATE"

git push --force invalidator sync/master-upstream
git push --force invalidator sync/master-local
```

Now for the `master` branch, `master-next` was used for the CI run from the
`origin` repo:

```sh
git push --force invalidator master-next:master
git push --force origin :master-next
```

Local versions of the sync branches are not needed any more:

```sh
git switch pr-branch
git branch --delete --force master-next
git branch --delete --force "sync/master/upstream/$SYNC_DATE"
git branch --delete --force sync/master-upstream
git branch --delete --force "sync/master/local/$SYNC_DATE"
git branch --delete --force sync/master-local
```

### 1.9. Unlock `master`

Now that `master` contains new changes it can be unlocked.

`pre-rebase` hook in individual user workspaces should prevent invalid rebase,
looking at the new `master` and the `sync/master/local/*` branch.

### 2. Late update

**NOTE** Normally, a `pre-rebase` hook will handle all the complexity, no matter
how long ago was the local `master` branch created.  Process below describes
synchronization in case when the `pre-rebase` hook is not used, for whatever
reason.

First, you need to look for closest `sync/master/local/[date]` branch in the
change history.  It would be your rebase baseline.  Anything after this point
should be your local changes.  While everything before is guaranteed to be
included in the `master`.

Run the following to find the right branch, if you do not want to use `git log`:

```sh
git switch your-pr

for base in $(
    git branch --list --no-column 'invalidator/sync/master/local/*' | sed -e 's@^\s*@@'
); do
    echo "$( git log --oneline $base..HEAD | wc -l ) $base";
done \
    | sort -n \
    | head -n 1 \
    | cut -d " " -f 2-
```

It will go though all the recorded `master` branch positions, looking for the
one that has the least number of changes compared to the currently checkout out
branch.

When you know your baseline, rebase your changes with

```
git rebase --onto master invalidator/sync/master/local/yyyy-mm-dd
```

Alternatively, you can run `scripts/pre-rebase-hook` to see the `git rebase`
command that you should run to rebase your changes on top of the latest `master`
correctly.

## Slow rebase

"Slow rebase" operation performed by `scripts/slow-rebase.sh` is, conceptually,
pretty simple.  It all boils down to the `run_one_rebase()` function, executed
in a loop.  `run_one_rebase()` runs the following rebase operation, with the
`master-next` branch checked out.

It looks at all changes that are in `sync/master-upstream` that are not in
`master-next` yet and takes the oldest of these changes.  It then rebases
`master-next` on top of this change.  Which bringing in the oldest commit from
`solana/master` into `master-next` that was not part of `master-next` yet.

If there are any merge conflicts between commits in `master-next` and
`solana/master`, this conflict will be presented in the smallest possible scope,
making conflict resolution easier.

`slow-rebase.sh` will compile, format and run a few more checks after this
single rebase step, again, reducing the context of any possible compilation
errors to a single upstream change.

And this process is repeated over and over, until all changes from
`sync/master-upstream` are added into `master-next`.

## On tags vs branches

Semantically, `sync/master/local/[date]` and `sync/master/upstream/[date]` look
more like tags than branches.  They are not supposed to move.

The reason they are branches are:

1. Branches retain their repository origin, while tags do not.

   This might not be very important, but it is nice to see what was the source
   of the synchronization information.

2. Branches can be updated, which could be useful if someone makes a mistake.

   While it is possible to update a tag position, everyone who already received
   the tag must run an explicit update for this particular tag.
   It matches what you want for releases in a decentralized system.
   It would probably only be a source of frustration in our setup.

These arguments are not bullet proof and the process could be updated to use
tags.  But, for now, the process works with branches and especially while the
process is still relatively new, the second point - an ability to fix a mistake,
seems like a very good property.

At the same time, there seems to be no immediate upside in using tags.

## Release branches

Release branches use the same approach as the master branch.  For example, for
`v1.16` the following branches are used:

 * `sync/v1.16/upstream/[date]`<br/>
   State of `v1.16` in the upstream.

 * `sync/v1.16/local/[date]`<br/>
   State of `v1.16` before the synchronization on the
   specified date.

 * `sync/v1.16-upstream`<br/>
   Latest `sync/v1.16/upstream/[date]`.

 * `sync/v1.16-local`<br/>
   Latest `sync/v1.16/local/[date]` branch.

 * `v1.16`<br/>
   All changes in `sync/v1.16-upstream`, plus all changes from
   `sync/v1.16-local`, plus all changes in `invalidator/v1.16` that were created
   since the last synchronization.

Synchronization process for `v1.16` is almost identical to the `master`
synchronization process, with `s/master/v1.16`.  Look above for a more detailed
explanation.  Here is just a list of commands you can copy/paste:

```sh
export SYNC_DATE=$(TZ=UTC date "+%Y-%m-%d")

git fetch upstream v1.16
git branch --no-track "sync/v1.16/upstream/$SYNC_DATE" \
    "$( TZ=UTC \
        git log --max-count=1 \
            --until="$SYNC_DATE 00:00:00+00:00" \
            --format=format:%H \
            upstream/v1.16 \
    )"

git branch --no-track sync/v1.16-upstream "sync/v1.16/upstream/$SYNC_DATE"

git fetch invalidator v1.16
git branch --no-track "sync/v1.16/local/$SYNC_DATE" invalidator/v1.16
git branch --no-track "sync/v1.16-local" "sync/v1.16/local/$SYNC_DATE"

git switch --create v1.16-next --no-track sync/v1.16-local
```

State of all branches that are involved in the `v1.16` sync process:

```sh
git log --oneline --decorate --graph \
    --branches='invalidator/sync/v1.16/upstream/*' \
    --branches='invalidator/sync/v1.16/local/*' \
    invalidator/sync/v1.16-upstream invalidator/sync/v1.16-local \
    sync/v1.16-upstream sync/v1.16-local \
    upstream/v1.16 \
    v1.16-next HEAD
```

Slow rebase local changes:

```sh
./scripts/slow-rebase.sh --branch v1.16 --date "$SYNC_DATE"
```

In case you created any "DO NOT SUBMIT: Fixup" commits:

```sh
git rebase --interactive \
    --reschedule-failed-exec \
    --exec "./cargo check --lib --bins --tests" \
    --exec "./scripts/cargo-fmt.sh" \
    --exec "cd programs/sbf \
        && ../../cargo check --bins --tests" \
    "$( git merge-base sync/v1.16-upstream HEAD )"
```

Run the CI:

```sh
git push origin --force v1.16-next
```

Publish the updates:

```sh
git push invalidator "sync/v1.16/upstream/$SYNC_DATE"
git push invalidator "sync/v1.16/local/$SYNC_DATE"

git push --force invalidator sync/v1.16-upstream
git push --force invalidator sync/v1.16-local

git push --force invalidator v1.16-next:v1.16
git push --force origin :v1.16-next

git switch pr-branch
git branch --delete --force v1.16-next
git branch --delete --force "sync/v1.16/upstream/$SYNC_DATE"
git branch --delete --force sync/v1.16-upstream
git branch --delete --force "sync/v1.16/local/$SYNC_DATE"
git branch --delete --force sync/v1.16-local
```
