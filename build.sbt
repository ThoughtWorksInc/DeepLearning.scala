organization := "com.qifun"

name := "immutable-future"

version := "0.1.1-SNAPSHOT"

libraryDependencies <+= (scalaVersion) { sv =>
  "org.scala-lang" % "scala-reflect" % sv
}

libraryDependencies += "junit" % "junit-dep" % "4.10" % "test"

libraryDependencies += "com.novocode" % "junit-interface" % "0.10" % "test"

scalacOptions ++= Seq("-optimize", "-deprecation", "-unchecked", "-Xlint", "-feature")

crossScalaVersions := Seq("2.10.4", "2.11.0-RC4")

description := "The rubost asynchronous programming facility for Scala that offers a direct API for working with Futures."

homepage := Some(url("https://github.com/Atry/immutable-future"))

startYear := Some(2014)

licenses := Seq("Apache License, Version 2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html"))

publishTo <<= (isSnapshot) { isSnapshot: Boolean =>
  if (isSnapshot)
    Some("snapshots" at "https://oss.sonatype.org/content/repositories/snapshots")
  else
    Some("releases" at "https://oss.sonatype.org/service/local/staging/deploy/maven2")
}

scmInfo := Some(ScmInfo(
  url("https://github.com/Atry/immutable-future"),
  "scm:git:git://github.com/Atry/immutable-future.git",
  Some("scm:git:git@github.com:Atry/immutable-future.git")))

pomExtra :=
  <developers>
    <developer>
      <id>Atry</id>
      <name>杨博</name>
      <timezone>+8</timezone>
      <email>pop.atry@gmail.com</email>
    </developer>
  </developers>
