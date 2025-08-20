# Synchronizing changes with the upstream

**NOTE** This document is targeted at someone actually merging changes from the
upstream.  If you are just using this repo for development, you do not
necessarily need to read it or to understand all the details.

We need to merge changes from the upstream repository into this repository
regularly, in order to reduce merge conflicts and to stay up to date.

We are going to call this process "upstream sync", to distinguish it from other
synchronization activities.

For now, we are only tracking the `master` branch of the upstream repository.
That would be `master` in `https://github.com/anza-xyz/agave`.  We used to track
`v1.16` as well and some notes are recorded at the ["Release
branches"](#release-branches) section.

`invalidator` uses the following branches to track state relative to the
upstream Anza repo.  Note that all branches with the `sync/` prefix are created
and/or updated only during the synchronization process.

 * `sync/master/upstream/[date]`<br/>
   Records state of the `master` branch in the upstream as of the specified
   date.  For example `sync/master/upstream/2023-04-03` would be one such
   branch.<br/>
   These are created for every synchronization.  And we keep them around in
   order to be able to check changes that happened as part of the upstream sync
   process.<br/>

 * `sync/master/local/[date]`<br/>
   Records state of the `master` branch before the synchronization on the
   specified date.<br/>
   These branches help rebase local changes and PRs, after `master` is
   rebased.<br/>
   The are used by the `scripts/pre-rebase-hook` logic.<br/>

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

# Synchronization process

Synchronization with the upstream happens in a few steps.

In order to avoid accidentally pushing `invalidator` changes into the public
repository it is strongly recommended that you use a dedicated workspace for
synchronization.  With remote names that would reduce this kind of push.  You
can create this setup as follows:

```sh
git clone --origin invalidator git@github.com:anza-xyz/invalidator.git
cd invalidator
git remote add upstream https://github.com/anza-xyz/agave.git
git remote set-url --push upstream do-not-push-from-invalidator-to-agave
```

Add your own invalidator fork as an `origin` remote - it will be used for the CI
runs.  If you have `gh` GitHub client then do this:

```sh
git remote add origin \
    git@github.com:"$( gh api user --jq .login )"/invalidator.git
```

Otherwise, replace the `gh` call with your GitHub user name.

Commands below assume `invalidator`, `upstream` and `origin` remotes are setup
as specified above.

```sh
git remote -v
```
```
invalidator     git@github.com:anza-xyz/invalidator.git (fetch)
invalidator     git@github.com:anza-xyz/invalidator.git (push)
origin	git@github.com:ilya-bobyr/invalidator.git (fetch)
origin	git@github.com:ilya-bobyr/invalidator.git (push)
upstream        https://github.com/anza-xyz/agave.git (fetch)
upstream        do-not-push-from-invalidator-to-agave (push)
```

## 1. Record update date.

The following steps are run by a single person.  Preferably on Monday.

All commands below assume you recorded the current date as:

```sh
export SYNC_DATE=$(TZ=UTC date "+%Y-%m-%d")
```

## 2. Process automation.

The synchronization process is quite complex.  Actions can be assessed based on
two orthogonal criteria: occurrence frequency and automation complexity.  It
makes sense to automate everything that happens frequently and is easy to
automate.  Things that are hard to automate but happen frequently are mostly
described in this doc.  Things that do not happen frequently might or might not
be described in this doc - and you still need to use your best judgement, as if
something happens infrequently, there could be some subtle details that were not
captured just yet.

Another aspect of the automation is that current automation is written in bash.
It is a very bad language for any non-trivial logic.  And I am not sure that
extending our bash codebase is wise.  We should consider writing a sync
automation tool in Rust, if we are going to extend our automation coverage with
any non-trivial logic.

For the moment, you'll need to copy/paste commands from this document and use
the `slow-rebase.sh` automation script.

## 3. Parallel updates in `master` and other invalidator branches

An execution of the upstream sync process takes from almost a whole day to
several days, depending on the complexity of the manual merges and how quickly a
person running the sync is reacting to the script stops.

Initially, we were considering an option of locking the `master` branch for the
duration of the sync process.  But as the process can take this long, it is
impractical.

Instead, we allow parallel changes to be submitted into the `invalidator/master`
branch, to be cherry picked into the updated `master` at the end of the sync
process.  This is described in detail in the ["Publish updated `master` and
`sync/*` branches"](#8.-Publish-updated-`master`-and-`sync/*`-branches)
section.

We may still want to lock the `master` branch for a short duration just while it
is being updated, to avoid any race conditions.  But, as git push shows the
branch SHA before it was updated.  Even if there was a parallel update, we will
see that and can recover any lost changes.  As the frequency of updates is
currently relatively low, recovery from a race condition is the trade-off we
have chosen for the moment.

## 4. Create a new sync point, and `master-next`.

We want to use a commit from `agave/master` that is the last change before
00:00 UTC on Monday.  Hopefully that would be a rather low activity point.

Fetch the latest state of the `https://github.com/anza-xyz/agave` `master`,
and point `sync/master/upstream/$SYNC_DATE` at the commit just before
`$SYNC_DATE` starts in UTC:

```sh
git fetch upstream master
git branch --no-track "sync/master/upstream/$SYNC_DATE" \
    "$( TZ=UTC \
        git log --max-count=1 \
            --until="$SYNC_DATE 00:00:00+00:00" \
            --format=format:%H \
            upstream/master \
    )"
```

Create all the other tracking branches, along with `master-next`:

```sh
git branch --no-track sync/master-upstream "sync/master/upstream/$SYNC_DATE"

git fetch invalidator master
git branch --no-track "sync/master/local/$SYNC_DATE" invalidator/master
git branch --no-track "sync/master-local" "sync/master/local/$SYNC_DATE"

git switch --create master-next --no-track sync/master-local
```

## 5. Apply history edits in the invalidator `master`

***TODO***  This section is incomplete at the moment.  You may skip it for now.

We are still working on the process of cleaning up the `invalidator` history, in
order to both simplify it, and to make conflicts easier to resolve.  As part of
this process `master` branch may contain `fixup!`, `squash!` and `amend!` tags
that `git rebase --autosquash` recognizes.  They should be applied, verifying
that the end state in `master-next` is identical to the `invalidator/master` it
started with, except for merge resolutions.

We might need to apply more complex history rewrites.  Those are marked with
`SQUASH`, `FIXUP` and `AMEND` tags in the commits titles.  These tags indicate
that an operation similar to the one automatically performed by `git
rebase` should be done, but the details need to be described in the
corresponding commit message.  These edits should be applied next.  Again, the
final state in `master-next` should not be different from `invalidator/master`,
except for merge resolutions.

## 6. Rebase `master-next` on top of the new sync point

Rebase changes on top of the updated upstream master.

As there are generally enough changes accumulated in a week, it is better to
rebase one upstream change at a time, compiling after every rebase.  This will
identify conflicts as soon as they arise, making it easier to resolve them.

In the workspace root, run:

```sh
./scripts/slow-rebase.sh --date "$SYNC_DATE"
```

The script will run the rebase, one upstream commit at a time, making sure that
the tip of the rebased `invalidator/master` compiles and is correctly formatted.
If anything fails the script should tell you what to do next and how to restart
the rebase process.

If you created any "DO NOT SUBMIT: Fixup" commits per script instructions, after
a complete rebase is done, I do another run like this:

```sh
git rebase --interactive \
    --reschedule-failed-exec \
    --exec "./cargo check --lib --bins --tests" \
    --exec "cd programs/sbf \
        && ../../cargo check --bins --tests" \
    --exec "./scripts/cargo-for-all-lock-files.sh -- \
        +$(bash -c 'source ci/rust-version.sh; echo $rust_nightly') fmt --all" \
    --exec "./scripts/cargo-clippy-nightly.sh" \
    "$( git merge-base sync/master-upstream HEAD )"
```

It redistributes the manually created fixups into the right commits, possibly
with some minor adjustments.

If you need to see relative state of the branches, run:

```sh
git log --graph --decorate --oneline \
    sync/master-upstream sync/master-local \
    master master-next HEAD \
    --not sync/master-upstream^
```

Or, if you want to see the full picture:

```sh
git log --graph --decorate --oneline \
    --remotes='invalidator/sync/master/upstream/*' \
    --remotes='invalidator/sync/master/local/*' \
    --branches='sync/master/upstream/*' \
    --branches='sync/master/local/*' \
    sync/master-upstream sync/master-local \
    upstream/master \
    master-next \
    invalidator/master HEAD
```

If you are trying to resolve a conflict and want to see a change in the upstream
that caused the conflict run

```sh
git show $( git rev-list master-next..sync/master-upstream | tail -n 1 )
```

If you want to be even more careful, you could run `cargo test` after every
merge, though, it would be very slow, as of today the `invalidator/master`
already contains 250+ commits:

```sh
./scripts/slow-rebase.sh --date "$SYNC_DATE" --run-tests
```

### 6.1. Frozen ABI hashes

If a change affects a type with an `AbiExample` instance, it may change the hash
value for this type.  A few changes in the `invalidator` repo touch types with
the `AbiExample` instances.  In particular, change that extracted
`solana-net-protocol` caused at least one hash to change.

You can wait for the CI to tell you if any of the hashes mismatch.

Alternatively, you can check it yourself either by running

```sh
./test-abi.sh
```

to check the whole repo.  Or you can run `cargo test` with the nightly compiler,
explicitly specifying the package, if you want to save some time:

```sh
./cargo nightly test --package solana-net-protocol --features frozen-abi --lib -- test_abi_
```

### 6.2. Updates of the third party dependency versions with multiple versions

When the invalidator branch introduces a new dependency on a third party crate,
it may create conflicts during the rebase.  While below I am describing a more
complex case, with multiple versions of `itertools` in the dependency graph, the
suggested resolution works quite well for simpler `Cargo.lock` updates as well.

For this example, let's assume that a dependency on `itertools` was added to the
`Cargo.toml` in one of the crates in the invaldiator repo:

```toml
[dependencies]
itertools = { workspace = true }
```

Let's say it was the `shred-dos` crate, so the above is then added to
`shred-dos/Cargo.toml`.  A new dependency on on `itertools` for the `shred-dos `
would crate a recorded in `Cargo.lock` using the current version of `itertools`.

Consider, what happens when `itertools` version is updated in the workspace
`Cargo.toml`.

If initially, the whole workspace was using just one version of `itertools`,
then all crates that use `itertools` will specify the dependency as just, well,
`itertools`, and the update would only touch the definition of `itertools`.
This case would not create any conflicts, and the dependency we introduced in
the invalidator repo will also be "automatically" updated.

But if there are multiple versions of `itertools` in the workspace, then the
situation becomes more complex.  In this case, `shred-dos` version of
`itertools` in `Cargo.lock` would be recorded with a version specifier, say as
`itertools 0.10.5`.

Now, if the upstream `itertools` is updated, the repo level `itertools` "name"
in the `Cargo.lock` will also change.  Yet, `git rebase` does not know that
`itertools 0.10.5` for `shred-dos` in `Cargo.lock` needs to be updated to as
well.  It might need to be changed to `itertools` without an explicit version,
or it might need to be updated to, say, `itertools 0.12.1`, which is the version
after the updated.

Besides introducing conflicts the above might also create updates for
`Crago.lock` files in the wrong commits.

One approach is to just recompile every commit in the invalidator `master`
branch.  But with 200+ commits and growing it is a very slow approach.

I found the following shortcut.  Run `git log` to see a full list of commits
where `itertools` was updated:

```bash
git log -Gitertools --oneline upstream/master..master-next -- \
    Cargo.lock programs/sbf/Cargo.lock
```

This list would often be quite short.  Now run an interactive rebase:

```bash
git rebase --interactive "$( git merge-base sync/master-upstream HEAD )"
```

Add the following `exec` checks before each commit that touches the updated
dependency:

```gitrebase
exec ./cargo check --lib --bins --tests
exec cd programs/sbf/ && ../../cargo check --bins --tests
exec ./scripts/cargo-for-all-lock-files.sh -- +"$(bash -c 'source ci/rust-version.sh; echo $rust_nightly')" fmt --all
exec ./scripts/cargo-clippy-nightly.sh
```

`cargo check` will update any versions to their up to date state and, as a
result, `Cargo.lock` updates will happen in the correct commits.  `git rebase`
will stop whenever `Cargo.lock` is updated due to the `cargo check` invocation,
and you just need to do the following when it stops:

```bash
git status
git diff

# Make sure that it is indeed only the `Cargo.lock` that is updated and that the
# updates are only for the dependency in question.

git add --update
git commit --amend
git rebase --continue
```

Unfortunately, you might be able to run the above process only *after* you
resolve conflicts.  So, just resolve them in any commit where they pop up.  And
the process above should shift `Cargo.lock` updates into the right commits when
you run it later.

## 7. Run CI for `master-next`

Create a PR with the `master-next` content in the `invalidator` repo and wait
for a successful CI result.

```sh
git push origin --force master-next
```

It is better to use consistent names for the CI PRs, just in case we want to
reference them later.  Here is the name pattern used so far:

```
`2024-08-23`: CI run for `master`
```

If you have the official GitHub CLI client, you can create a PR with this name
like this:

```sh
gh pr create \
    --title "\`$SYNC_DATE\`: CI run for \`master\`" \
    --body '' \
    --no-maintainer-edit \
    --draft \
    --repo $( git remote -v get-url invalidator \
        | sed -e 's/^[^:]*://; s/\.git$//' ) \
    --base master \
    --head $( gh api user --jq .login ):master-next
```

```fish
gh pr create \
    --title "`$SYNC_DATE`: CI run for `master`" \
    --body '' \
    --no-maintainer-edit \
    --draft \
    --repo $( git remote -v get-url invalidator \
        | sed -e 's/^[^:]*://; s/\.git$//' ) \
    --base master \
    --head $( gh api user --jq .login ):master-next
```

## 8. Publish updated `master` and `sync/*` branches

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
```

Look at the output from the above command.  It will list SHA that `master` was
at and the new `SHA` it was updated to.  Old SHA should match your
`sync/master-local`.  If it does not, then it means that someone has submitted a
change while you were preparing the sync.  You need to cherry pick this change
on top of the `master-next` and redo parts of the above.

***TODO*** Describe the exact steps of this correction.  It is actually better
to check if `invalidator/master` has any updates just before the `push --force`
to reduce the risk of a parallel update.

Finally, remove `master-next` from GitHub, it was only needed for the CI to run
and to record the update in form of a PR:

```sh
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

# Late update

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
git switch your-pr-branch

git branch --list --remote --no-column 'invalidator/sync/master/local/*' \
    | sed -e 's@^\s*@@' \
    | while read -e base; do
        echo "$( git log --oneline $base..HEAD | wc -l ) $base";
      done \
    | sort -n \
    | head -n 1 \
    | cut -d " " -f 2-
```

```fish
git switch your-pr-branch

git branch --list --remote --no-column 'invalidator/sync/master/local/*' \
    | sed -e 's@^\s*@@' \
    | while read --local base
        echo ( git log --oneline $base..HEAD | wc -l ) $base
      end \
    | sort -n \
    | head -n 1 \
    | cut -d " " -f 2-
```

It will go though all the recorded `master` branch positions, looking for the
one that has the least number of changes compared to the currently checkout out
branch.

When you know your baseline, rebase your changes by running:

```
git rebase --onto master invalidator/sync/master/local/yyyy-mm-dd
```

Alternatively, you can run `scripts/pre-rebase-hook` to see the `git rebase`
command that you should run to rebase your changes on top of the latest `master`
correctly.

# Slow rebase

"Slow rebase" operation performed by `scripts/slow-rebase.sh` is, conceptually,
pretty simple.  It all boils down to the `run_one_rebase()` function, executed
in a loop.  `run_one_rebase()` runs the following rebase operation, with the
`master-next` branch checked out.

It looks at all changes that are in `sync/master-upstream` that are not in
`master-next` yet and takes the oldest of these changes.  It then rebases
`master-next` on top of this change.  Which bringing in the oldest commit from
`agave/master` into `master-next` that was not part of `master-next` just yet.

If there are any merge conflicts between commits in `master-next` and
`agave/master`, this conflict will be presented in the smallest possible scope,
making conflict resolution easier.

`slow-rebase.sh` will run the compilation, source formatting and run a few more
checks after this single rebase step, again, reducing the context of any
possible compilation errors to a single upstream change.

And this process is repeated over and over, until all changes from
`sync/master-upstream` are added into `master-next`.

`slow-rebase.sh` is also trying to show some meaningful error messages in case
it cannot continue automatically, in order to help the person running the merge
perform the next step.

# Compiling every commit on the `invalidator` branch

Our process only compiles the very last commit on the `invalidator/master`
branch, even though we might use `git absorb` to modify intermediate commits, as
well as do merge resolutions on intermediate commits.

Ideally, we would want to compile and format every commit from the
`invalidator/master` branch as it is applied, rather than doing all the checks
for the branch as a whole, after everything is rebased on top of the next
upstream commit.  This way, the compilation or formatting is immediately
attributed to the right commit in the `master-next` branch.  But, is too slow.

It is possible for the commits in the `invalidator/master` branch to become
broken without the final state being broken.  And it would be a problem if a
merge conflict happens later in a commit that was incorrect.

To avoid these situations it would probably be good to run compilation for all
commits in the `invalidator/master` branch from time to time:

```sh
git rebase --interactive \
    --reschedule-failed-exec \
    --exec "./cargo check --lib --bins --tests" \
    --exec "cd programs/sbf \
        && ../../cargo check --bins --tests" \
    --exec "./scripts/cargo-for-all-lock-files.sh -- \
        +$(bash -c 'source ci/rust-version.sh; echo $rust_nightly') fmt --all" \
    --exec "./scripts/cargo-clippy-nightly.sh" \
    "$( git merge-base sync/master-upstream HEAD )"
```

Maybe we could have a CI action that does this automatically, outside of the
normal update sync flow, to save developer time.

# On tags vs branches

Semantically, `sync/master/local/[date]` and `sync/master/upstream/[date]` look
more like tags than branches.  They are not supposed to move.

The reason they are branches are:

1. Branches retain their repository origin, while tags do not.

   This might not be very important, but it is nice to see what was the source
   of the synchronization information.

2. Branches can be updated, which could be useful if someone makes a mistake.

   While it is possible to update a tag position, everyone who has already
   received the tag must run an explicit update command for this particular tag.
   This is a good behaviour for marking releases in a decentralized system.
   It would probably only be a source of frustration in our setup.

These arguments are not bullet proof and the process could be updated to use
tags.  But for now, the process works with branches.  And in particular, as we
are still refining the sync process and as a considerable number of steps are
performed by humans an ability to fix a mistake is very handy.  I've incorrectly
marked sync branches at least a few times since the sync process started.

And there seems to be no immediate upside in using tags.

# Release branches

We are not keeping any release branches up to date at the moment.  Process for
`v1.16` is described below.  Let's try to keep it up to date as a best effort.

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
synchronization process, with `s/master/v1.16/g`.  Look above for a more
detailed explanation.  Here is just a list of commands you can copy/paste:

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
    --exec "cd programs/sbf \
        && ../../cargo check --bins --tests" \
    --exec "./scripts/cargo-for-all-lock-files.sh -- \
        +$(bash -c 'source ci/rust-version.sh; echo $rust_nightly') fmt --all" \
    --exec "./scripts/cargo-clippy-nightly.sh" \
    "$( git merge-base sync/v1.16-upstream HEAD )"
```

Run the CI:

```sh
git push origin --force v1.16-next
```

```sh
gh pr create \
    --title "\`$SYNC_DATE\`: CI run for \`v1.16-next\`" \
    --body '' \
    --no-maintainer-edit \
    --draft \
    --repo $( git remote -v get-url invalidator \
        | sed -e 's/^[^:]*://; s/\.git$//' ) \
    --base v1.16 \
    --head $( gh api user --jq .login ):v1.16-next
```

```fish
gh pr create \
    --title "`$SYNC_DATE`: CI run for `v1.16-next`" \
    --body '' \
    --no-maintainer-edit \
    --draft \
    --repo $( git remote -v get-url invalidator \
        | sed -e 's/^[^:]*://; s/\.git$//' ) \
    --base v1.16 \
    --head $( gh api user --jq .login ):v1.16-next
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
