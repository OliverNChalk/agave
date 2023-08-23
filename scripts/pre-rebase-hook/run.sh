#!/usr/bin/env bash

# Implementation of the `pre-rebase` hook to be run by the user workspace `git
# rebase` command.
#
# It is designed to prevent accidental rebases of duplicate history, as the
# `invalidator` repo has the `master` branch rebased.
#
# See `../../docs/upstream-sync.md` for details.

# Repository wide configuration

syncPrefix=$(git config --get --default sync invalidator-repo.sync-prefix)
upstreamRemote=$(git config --get --default '' invalidator-repo.upstream)


# ----
#
# Implementation details.
#
# Git will call this hook with 2 parameters:
#
# $1 - the upstream the series was forked from.
# $2 - the branch being rebased (or empty when rebasing the current branch).
#
# Essentially, "$1..$2" is the range of commits that are about to be rebased.
# Git does not pass the destination of the rebase into the hook.
#
# As git branches are not containers, even if a destination commit will be
# provided, it would still not be impossible to know for sure when target of a
# rebase is the `master` branch.  It is probably possible to check if the rebase
# target is the tip of the `upstream/master` branch, for example.  But `git
# rebase` does not provide this information.  So even getting it is not
# immediately trivial.
#
# One way I was able to come up with would be to look at the parent process,
# which would be the `git rebase` invocation, and parse the arguments.  Looking
# for `--onto upstream/master` or something like that.
#
# For the starters, I've decided to use what `git rebase` is already provides
# and set the constrains as follows.
#
# When `master` is rebased, the rebase operation should not include any changes
# that are part of any of the `upstream/sync/master/local` branches.  Only
# changes after that could be rebased.
#
# It seems safe and it is still possible to distinguish the case when the local
# `master` branch needs to rebased across the upstream rebase.  And provide a
# helpful message when user is not specifying the right rebase base commit.

set -e

# Set by `findLocalBaseline`. Holds `[upstream]/master` or
# `[upstream]/sync/master/local/[date]` - a branch that contains the most
# changes from `$branch`.
localBaseline=

base="$1"
if test "$#" -ge 2
then
  branch="refs/heads/$2"
else
  branch=$( git symbolic-ref HEAD ) || \
    exit 0 ; # Exit for detached HEAD.
fi

# A list of branches that a remote needs to contain, to be usable as a
# synchronization reference point.
declare -a remoteSyncBranches=(
    master \
    "$syncPrefix/master-upstream"
    "$syncPrefix/master/upstream/*"
    "$syncPrefix/master-local"
    "$syncPrefix/master/local/*"
  )

# We need to know which remote contains the synchronization branches, but it can
# have any name.  While we allow users to configure it as `[invalidator-repo]
# upstream`, we also want the hook to "just work".  And so, if not configured,
# we search among all the remotes, hoping to find a single match.
findInvalidatorUpstreamRemote() {
  local -a remotes=()
  while IFS='' read -r line; do remotes+=("$line"); done \
    < <( git remote show -n  )

  local candidate=
  local remoteBranch=
  local missing=
  for remote in "${remotes[@]}"; do
    missing=
    for name in "${remoteSyncBranches[@]}"; do
      remoteBranch="$remote/$name"
      if [[ "$( \
        git branch --list --remotes --no-column --no-color -- "$remoteBranch"
      )" = "" ]]; then
        missing=t
        break
      fi
    done

    if [[ -n "$missing" ]]; then
      continue
    fi

    if [[ -z "$upstreamRemote" ]]; then
      # User did not select a desired remote via a configuration parameter.
      if [[ -n "$candidate" ]]; then
        cat <<EOM
ERROR:
    Could not determine a unique upstream remote for the "invalidator" repo.
    At least two candidates found: "$candidate" and "$remote"
    Remove one of these remotes or select the desired one with

    git config --local invalidator-repo.upstream "$candidate"

    or

    git config --local invalidator-repo.upstream "$remote"
EOM
        exit 1
      fi
      candidate=$remote
    else
      candidate=$remote
      # We ignore all other remotes except the selected one.  But we still
      # want to make sure it contains a suitable set of branches.
      if [[ "$upstreamRemote" = "$remote" ]]; then
        break
      fi
    fi
  done

  if [[ -z "$candidate" ]]; then
    if [[ -z "$upstreamRemote" ]]; then
      cat <<EOM
ERROR:
    Could not find a remote that holds synchronization branches needed to track
    the "invalidator" repository \`master\` branch rebases.  Can not make sure
    your rebase is safe.

    A remote needs to contain the following branches:

EOM
      printf '    %s\n' "${remoteSyncBranches[@]}"
      cat <<EOM

    You can add an upstream pointing at the invalidator repo like this:

    git remote add upstream git@github.com:solana/invalidator.git
    git fetch upstream

    Or you can remove the rebase hook, to disable checking altogether:

    rm .git/hook/pre-rebase
EOM
      # If the hook is there, it means the user wanted rebase to be safe.
      exit 1
    else
      cat <<EOM
ERROR:
    You have an upstream remote configured as "$upstreamRemote", but it does not
    contain all the necessary tracking branches.  Remote needs to contain the
    following branches:

EOM
      printf '    %s\n' "${remoteSyncBranches[@]}"
      cat <<EOM

    You can remove the local configuration:

    git config --local --unset invalidator-repo.upstream

    Fix the remote or remove the hook to avoid this message:

    rm .git/hook/pre-rebase
EOM
      # Be conservative here.  If the user has configured a remote, assume that
      # they do want proper rebase protection.
      exit 1
    fi
  fi

  if [[ -n "$upstreamRemote" && "$upstreamRemote" != "$candidate" ]]; then
    cat <<EOM
ERROR:
    Upstream remote you have selected in your local workspace configuration does
    not contain the synchronization branches necessary to track "invalidator"
    repository \`master\` branch rebases.

    Selected remote: $upstreamRemote
    Found remote that contains tracking branches: $candidate

    You can remove local configuration:

    git config --local --unset invalidator-repo.upstream

    or update it:

    git config --local invalidator-repo.upstream "$candidate"
EOM
    exit 1
  fi

  upstreamRemote=$candidate
}

# Looks though all `[upstream]/sync/master/local/[date]` and `[upstream]/master"
# branches, looking for one that contains the most changes from "$branch".
#
# Sets "$localBaseline" to be equal to the found branch name.
findLocalBaseline() {
  local -a candidates=( "$upstreamRemote/master" )

  while IFS='' read -r line; do candidates+=("$line"); done < <(
    git branch --list --remotes --no-column --no-color \
        "$upstreamRemote/$syncPrefix/master/local/*" \
      | sed --expression='
          # Remove indentation
          s@^\s*@@
        ' \
      | sort -u
  )

  local baseline
  local reference
  baseline=$(
    for reference in "${candidates[@]}"; do
      echo "$( \
        git rev-list --pretty=oneline "$reference".."$branch" \
        | wc -l ) $reference"
    done \
      | sort -n \
      | head -n 1 \
      | cut -d " " -f 2-
  )

  localBaseline=$baseline
}

# Check if the rebased commits are contained in any of the `sync/master/local/*`
# branches.
#
# If contained only in the latest branch, suggest a better base for the rebase
# command.
#
# If contained in an older branch, inform that an upstream rebased had happened,
# as suggest a rebase command that includes a base.
#
# TODO Check the parent command arguments, for better suggestions in the cases
# above.
preventInvalidRebase() {
  rebaseList=$( git rev-list --pretty=oneline "$branch" ^"$base" )
  rebaseListButMasterLocal=$( \
      git rev-list --pretty=oneline "$branch" ^"$base" ^"$localBaseline" \
    )

  if [[ "$rebaseList" = "$rebaseListButMasterLocal" ]]; then
    # Rebase operation does not contain any changes that were already included
    # in the `[upstream]/master`.  We should allow this rebase.  It could be
    # completely unrelated to the `[upstream]/master in the first place.
    return
  fi

  # This rebase attempt contains changes that are already in
  # `[upstream]/master`.  We are not going to allow it.  The only question is:
  # what is the most helpful message that we can show?

  # TODO It is possible to show a better suggestion, if we use arguments that
  # were passed to the `git rebase` command itself.  I was able to get them by
  # looking at the `/proc/$PPID/cmdline`.  But we would need to parse them, as
  # they are stored as a string.  And then distinguish several different
  # invocations of `git rebase`.  So while possible, it is a considerable
  # additional effort.  Let's see if anyone will complain that the suggestion is
  # not good enough?

  if [[ "$localBaseline" = "$upstreamRemote/master" ]]; then
    cat <<EOM
ERROR:
  You are trying to rebase the following changes that are already part of the
  "$upstreamRemote/master" branch:

EOM
    git log --pretty=oneline \
      "$( git merge-base "$localBaseline" "$branch" )" ^"$base"

    cat <<EOM

  Consider using using an explicit baseline in your \`git rebase\` invocation:

  git rebase "$upstreamRemote/master" "$branch"

  (If you really know what you are doing, you can skip with check with a
  \`--no-verify\` argument.)
EOM
  else
    cat <<EOM
ERROR:
  You are trying to rebase the following changes that are already part of the
  "$upstreamRemote/master" branch:

EOM
    git log --pretty=oneline \
      "$( git merge-base "$localBaseline" "$branch" )" ^"$base"

    cat <<EOM

  Since the last time you rebased your changes on top of the
  "$upstreamRemote/master" branch, it has been rebased, to include changes from
  the upstream "solana" repo.  This could be the reason for this rebase to
  contain unnecessary changes.

  You probably want to rebase your work on top of the new
  "$upstreamRemote/master", before you continue any work, like this:

  # Rebase, excluding changes already in the "master" branch.
  git rebase --onto="$upstreamRemote/master" "$localBaseline"

  (If you really know what you are doing, you can skip with check with a
  \`--no-verify\` argument.)
EOM
  fi

  exit 1
}


# === Main sequence ===

findInvalidatorUpstreamRemote

findLocalBaseline

preventInvalidRebase
